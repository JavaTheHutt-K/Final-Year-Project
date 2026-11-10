"""Create dataset and experiment summaries without uploading or modifying source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_dataset(
    dataset_path: str | Path,
    *,
    label_column: str = "Attack Type",
    chunksize: int = 50_000,
) -> dict[str, Any]:
    """Summarize a CSV in chunks so the full dataset need not fit in memory."""
    path = Path(dataset_path)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    rows = 0
    duplicate_rows = 0
    seen_row_hashes: set[int] = set()
    missing_values: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    numeric_columns: list[str] | None = None
    numeric_sums: dict[str, float] = {}
    numeric_squared_sums: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}

    for frame in pd.read_csv(path, chunksize=chunksize):
        rows += len(frame)
        row_hashes = pd.util.hash_pandas_object(frame, index=False)
        duplicate_rows += sum(int(row_hash) in seen_row_hashes for row_hash in row_hashes)
        seen_row_hashes.update(int(row_hash) for row_hash in row_hashes)
        numeric = frame.select_dtypes(include="number")
        if numeric_columns is None:
            numeric_columns = list(numeric.columns)
        for column, count in frame.isna().sum().items():
            missing_values[column] = missing_values.get(column, 0) + int(count)
        if label_column in frame:
            for label, count in frame[label_column].value_counts(dropna=False).items():
                key = "<missing>" if pd.isna(label) else str(label)
                class_counts[key] = class_counts.get(key, 0) + int(count)
        for column in numeric_columns:
            if column not in numeric:
                continue
            values = numeric[column].dropna().astype(float)
            numeric_sums[column] = numeric_sums.get(column, 0.0) + float(values.sum())
            numeric_squared_sums[column] = (
                numeric_squared_sums.get(column, 0.0) + float((values * values).sum())
            )
            numeric_counts[column] = numeric_counts.get(column, 0) + int(values.size)

    if rows == 0:
        raise ValueError("dataset must contain at least one row")

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for column in numeric_columns or []:
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
        "features": len(numeric_columns or []),
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
