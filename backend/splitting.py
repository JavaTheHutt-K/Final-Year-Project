"""Reproducible, stratified partitions for a combined traffic dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_random(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create disjoint initial, update, and held-out partitions.

    The split preserves the binary class ratio in each partition. Since the
    lightweight combined dataset has no capture-day metadata, these partitions
    are not temporal and must not be interpreted as newer traffic.
    """
    dataset_config = config["dataset"]
    split_config = config["split"]
    if split_config.get("strategy") != "stratified_random":
        raise ValueError("split.strategy must be 'stratified_random'")
    label_column = dataset_config["binary_label_column"]
    if label_column not in data:
        raise ValueError(f"Data is missing configured label column {label_column!r}")

    initial_fraction = float(split_config["initial_train_fraction"])
    update_fraction = float(split_config["update_fraction"])
    test_fraction = float(split_config["test_fraction"])
    fractions = (initial_fraction, update_fraction, test_fraction)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Each split fraction must be greater than 0 and less than 1")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Initial, update, and test fractions must sum to 1")

    labels = data[label_column]
    if labels.nunique() != 2:
        raise ValueError("Stratified splitting requires both binary classes in the dataset")
    initial_count = int(round(len(data) * initial_fraction))
    test_count = int(round(len(data) * test_fraction))
    remainder_count = len(data) - initial_count
    update_count = remainder_count - test_count
    if min(initial_count, update_count, test_count) < 1:
        raise ValueError("The configured fractions produce an empty data partition")

    indices = data.index.to_numpy()
    initial_indices, remainder_indices = train_test_split(
        indices,
        test_size=remainder_count,
        random_state=int(config["training"]["random_seed"]),
        stratify=labels.to_numpy(),
    )
    remainder_labels = data.loc[remainder_indices, label_column].to_numpy()
    update_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=test_count,
        random_state=int(config["training"]["random_seed"]) + 1,
        stratify=remainder_labels,
    )
    if len(update_indices) != update_count:
        raise RuntimeError("The split library returned an unexpected update partition size")

    initial = data.loc[initial_indices].copy().reset_index(drop=True)
    update = data.loc[update_indices].copy().reset_index(drop=True)
    test_name = str(split_config["test_set_name"])
    tests = {
        test_name: data.loc[test_indices].copy().reset_index(drop=True),
    }

    if len(initial) + len(update) + len(tests[test_name]) != len(data):
        raise RuntimeError("Random partitions do not cover the prepared dataset")
    return initial, update, tests
