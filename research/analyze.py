"""Create dataset and experiment summaries without uploading or modifying source data."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd


DuplicateDetection = Literal["exact", "none"]


def analyze_dataset(
    dataset_path: str | Path,
    *,
    label_column: str = "Attack Type",
    chunksize: int = 50_000,
    duplicate_detection: DuplicateDetection = "exact",
) -> dict[str, Any]:
    """Summarize a CSV in chunks without retaining the dataset in memory.

    CSV fields are initially read as strings so type inference cannot vary
    between chunks. Numeric summaries coerce each field independently and
    include every value that is numerically parseable. Exact duplicate
    detection uses a temporary SQLite index on disk, so its storage grows with
    the number of distinct rows but Python memory stays bounded. Use
    ``duplicate_detection="none"`` to skip that disk usage.
    """
    path = Path(dataset_path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if duplicate_detection not in {"exact", "none"}:
        raise ValueError("duplicate_detection must be 'exact' or 'none'")

    rows = 0
    duplicate_rows = 0
    missing_values: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    numeric_columns: set[str] = set()
    numeric_sums: dict[str, float] = {}
    numeric_squared_sums: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}

    header = pd.read_csv(path, nrows=0)
    if label_column not in header.columns:
        raise ValueError(f"label column not found: {label_column}")

    with tempfile.TemporaryDirectory(prefix="ids-duplicate-index-") as temp_dir:
        connection: sqlite3.Connection | None = None
        if duplicate_detection == "exact":
            connection = sqlite3.connect(Path(temp_dir) / "rows.sqlite3")
            connection.execute(
                "CREATE TABLE rows (row_key BLOB PRIMARY KEY)"
            )

        try:
            for frame in pd.read_csv(
                path,
                chunksize=chunksize,
                dtype=str,
                keep_default_na=True,
            ):
                rows += len(frame)
                if connection is not None:
                    for row in frame.itertuples(index=False, name=None):
                        row_key = json.dumps(
                            [
                                None if pd.isna(value) else str(value)
                                for value in row
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        inserted = connection.execute(
                            "INSERT OR IGNORE INTO rows (row_key) VALUES (?)",
                            (row_key,),
                        ).rowcount
                        duplicate_rows += int(inserted == 0)
                    connection.commit()

                for column, count in frame.isna().sum().items():
                    missing_values[column] = missing_values.get(column, 0) + int(count)
                for label, count in frame[label_column].value_counts(
                    dropna=False
                ).items():
                    key = "<missing>" if pd.isna(label) else str(label)
                    class_counts[key] = class_counts.get(key, 0) + int(count)
                for column in frame.columns:
                    values = pd.to_numeric(frame[column], errors="coerce").dropna()
                    if values.empty:
                        continue
                    numeric_columns.add(column)
                    numeric_values = values.to_numpy(dtype=float)
                    numeric_sums[column] = numeric_sums.get(column, 0.0) + float(
                        numeric_values.sum()
                    )
                    numeric_squared_sums[column] = (
                        numeric_squared_sums.get(column, 0.0)
                        + float((numeric_values * numeric_values).sum())
                    )
                    numeric_counts[column] = numeric_counts.get(column, 0) + int(
                        numeric_values.size
                    )
        finally:
            if connection is not None:
                connection.close()

    if rows == 0:
        raise ValueError("dataset must contain at least one row")

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for column in sorted(numeric_columns):
        count = numeric_counts.get(column, 0)
        if not count:
            numeric_summary[column] = {"mean": None, "std": None}
            continue
        mean = numeric_sums[column] / count
        variance = max(numeric_squared_sums[column] / count - mean * mean, 0.0)
        numeric_summary[column] = {"mean": mean, "std": variance**0.5}

    return {
        "dataset": str(path),
        "rows": rows,
        "columns": len(pd.read_csv(path, nrows=0).columns),
        "features": len(numeric_columns),
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "class_distribution": class_counts,
        "numeric_summary": numeric_summary,
    }


def summarize_experiment(results_path: str | Path) -> dict[str, Any]:
    """Extract measured comparison metrics from the existing results JSON."""
    with Path(results_path).open(encoding="utf-8") as file:
        results = json.load(file)
    metrics = results.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("results JSON does not contain a metrics object")
    return {
        "results": str(results_path),
        "models": metrics,
        "training_history_rounds": len(results.get("training_history", [])),
        "evaluation": results.get("evaluation", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("path", type=Path)
    dataset_parser.add_argument("--label-column", default="Attack Type")
    dataset_parser.add_argument("--chunksize", type=int, default=50_000)
    dataset_parser.add_argument(
        "--duplicate-detection",
        choices=("exact", "none"),
        default="exact",
        help="Use an exact temporary disk index or skip duplicate detection",
    )
    dataset_parser.add_argument("--output", type=Path)
    results_parser = subparsers.add_parser("results")
    results_parser.add_argument("path", type=Path)
    results_parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "dataset":
        summary = analyze_dataset(
            arguments.path,
            label_column=arguments.label_column,
            chunksize=arguments.chunksize,
            duplicate_detection=arguments.duplicate_detection,
        )
    else:
        summary = summarize_experiment(arguments.path)
    serialized = json.dumps(summary, indent=2)
    if arguments.output:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
