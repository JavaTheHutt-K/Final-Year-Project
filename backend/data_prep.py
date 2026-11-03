"""Load and clean the lightweight combined CICIDS2017 traffic dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project settings from the root YAML configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def _resolve_path(root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _resolve_output_path(root: Path, configured_path: str) -> Path:
    """Resolve an output path, honoring the container's temporary output dir."""
    results_dir = os.environ.get("IDS_RESULTS_DIR")
    if results_dir:
        return Path(results_dir) / Path(configured_path).name
    return _resolve_path(root, configured_path)


def _resolve_output_dir(root: Path, configured_path: str) -> Path:
    """Resolve the output directory, honoring the container's temporary dir."""
    results_dir = os.environ.get("IDS_RESULTS_DIR")
    if results_dir:
        return Path(results_dir)
    return _resolve_path(root, configured_path)


def _clean_chunk(
    frame: pd.DataFrame,
    label_column: str,
    benign_label: str,
    attack_label_column: str,
    binary_label_column: str,
    drop_nan_rows: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one CSV chunk while retaining original and binary labels."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Column names are duplicated after whitespace cleanup")
    if label_column not in frame.columns:
        raise ValueError(
            f"Configured label column {label_column!r} is missing from the CSV"
        )
    if {attack_label_column, binary_label_column} & set(frame.columns):
        raise ValueError("Configured output label columns conflict with input columns")

    counts = {"before": len(frame)}
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    counts["after_infinity_replacement"] = len(frame)
    frame[label_column] = frame[label_column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    frame[label_column] = frame[label_column].replace("", np.nan)
    counts["after_label_cleanup"] = len(frame)
    if drop_nan_rows:
        frame = frame.dropna()
    counts["after_nan_rows"] = len(frame)

    labels = frame[label_column].astype(str)
    frame[attack_label_column] = labels
    frame[binary_label_column] = (
        labels.str.casefold() != benign_label.casefold()
    ).astype("int8")
    return frame.reset_index(drop=True), counts


def save_data_report(report: dict[str, Any], config_path: str | Path) -> None:
    """Write the cleaning and partition report to its configured result path."""
    config = load_config(config_path)
    root = Path(config_path).resolve().parent
    report_path = _resolve_output_path(root, config["paths"]["data_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


def prepare_data(
    config_path: str | Path = "config.yaml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, clean, and optionally sample the configured combined CSV in chunks.

    Chunk-level sampling limits peak memory for the lightweight file. Model
    partitions and scaling are handled later to avoid preprocessing leakage.
    """
    config = load_config(config_path)
    project_root = Path(config_path).resolve().parent
    dataset_config = config["dataset"]
    preprocessing_config = config["preprocessing"]
    raw_dir = _resolve_path(
        project_root,
        os.environ.get("IDS_DATA_DIR", config["paths"]["raw_data_dir"]),
    )
    csv_path = _resolve_path(raw_dir, dataset_config["source_file"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"Configured dataset file was not found: {csv_path}")

    label_column = str(dataset_config["label_column"]).strip()
    attack_label_column = str(dataset_config["attack_label_column"])
    binary_label_column = str(dataset_config["binary_label_column"])
    benign_label = str(dataset_config["benign_label"]).strip()
    sample_fraction_value = dataset_config["sample_fraction"]
    if isinstance(sample_fraction_value, bool):
        raise ValueError("dataset.sample_fraction must be a finite number between 0 and 1")
    sample_fraction = float(sample_fraction_value)
    chunk_size = int(dataset_config["chunk_size"])
    if not np.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
        raise ValueError("dataset.sample_fraction must be greater than 0 and at most 1")
    if chunk_size < 1:
        raise ValueError("dataset.chunk_size must be a positive integer")

    seed = int(config["training"]["random_seed"])
    sampled_chunks: list[pd.DataFrame] = []
    seen_row_hashes: set[int] = set()
    row_totals = {
        "before": 0,
        "after_infinity_replacement": 0,
        "after_label_cleanup": 0,
        "after_nan_rows": 0,
        "after_duplicate_rows": 0,
    }
    for chunk_number, frame in enumerate(
        pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
    ):
        cleaned, counts = _clean_chunk(
            frame,
            label_column,
            benign_label,
            attack_label_column,
            binary_label_column,
            bool(preprocessing_config["drop_nan_rows"]),
        )
        for stage, count in counts.items():
            row_totals[stage] += count
        if preprocessing_config["drop_duplicate_rows"]:
            row_hashes = pd.util.hash_pandas_object(
                cleaned,
                index=False,
            ).to_numpy(dtype=np.uint64)
            keep_rows = []
            for row_hash in row_hashes:
                hash_value = int(row_hash)
                keep_rows.append(hash_value not in seen_row_hashes)
                seen_row_hashes.add(hash_value)
            cleaned = cleaned.loc[keep_rows]
        row_totals["after_duplicate_rows"] += len(cleaned)
        if sample_fraction < 1:
            cleaned = cleaned.sample(
                frac=sample_fraction,
                random_state=seed + chunk_number,
            )
        if not cleaned.empty:
            sampled_chunks.append(cleaned)

    if not sampled_chunks:
        raise ValueError("No rows remain after dataset cleaning and sampling")
    data = pd.concat(sampled_chunks, ignore_index=True, sort=False)
    data = data.reset_index(drop=True)
    sampled_rows = len(data)

    protected_columns = {
        label_column,
        attack_label_column,
        binary_label_column,
    }
    numeric_features = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in protected_columns
    ]
    report: dict[str, Any] = {
        "config": {
            "source_file": str(csv_path),
            "label_column": label_column,
            "benign_label": benign_label,
            "sample_fraction": sample_fraction,
            "chunk_size": chunk_size,
        },
        "sources": {
            csv_path.name: {
                "files": [str(csv_path)],
                "rows": {
                    **row_totals,
                    "after_sampling": sampled_rows,
                },
                "class_balance": {
                    "benign": int((data[binary_label_column] == 0).sum()),
                    "malicious": int((data[binary_label_column] == 1).sum()),
                },
                "attack_label_counts": {
                    str(label): int(count)
                    for label, count in data[attack_label_column]
                    .value_counts()
                    .items()
                },
            }
        },
        "partitions": {},
        "constant_columns_dropped": [],
        "totals": {
            "rows_before_cleaning": row_totals["before"],
            "rows_after_cleaning": int(len(data)),
            "columns_after_cleaning": int(len(data.columns)),
            "feature_columns_after_cleaning": len(numeric_features),
            "constant_columns_dropped": 0,
        },
    }
    save_data_report(report, config_path)
    return data, report


def drop_training_constant_columns(
    data: pd.DataFrame,
    training_data: pd.DataFrame,
    excluded_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features constant in initial training rows, not unseen holdout rows."""
    constant_columns = [
        column
        for column in training_data.columns
        if column not in excluded_columns
        and training_data[column].nunique(dropna=False) <= 1
    ]
    return data.drop(columns=constant_columns), constant_columns


def fit_feature_scaler(
    training_data: pd.DataFrame,
    excluded_columns: list[str],
    scaler_name: str = "standard",
) -> tuple[StandardScaler, list[str]]:
    """Fit a scaler using training data only and return its numeric features."""
    if scaler_name != "standard":
        raise ValueError(f"Unsupported scaler configured: {scaler_name}")
    feature_columns = [
        column
        for column in training_data.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available to scale")
    scaler = StandardScaler()
    scaler.fit(training_data[feature_columns])
    return scaler, feature_columns


def transform_features(
    data: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform data with a scaler previously fitted on training rows."""
    missing_columns = set(feature_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Data is missing scaler feature columns: {sorted(missing_columns)}")
    transformed = data.astype({column: np.float64 for column in feature_columns})
    transformed[feature_columns] = scaler.transform(data[feature_columns])
    return transformed
