"""Evaluation helpers for the intrusion detection study."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate binary metrics with malicious traffic as the positive class."""
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Binary labels and predictions must contain only 0 and 1")

    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    negative_count = int(tn + fp)
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1_score": float(f1_score(truth, prediction, zero_division=0)),
        "false_positive_rate": float(fp / negative_count) if negative_count else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_labels: np.ndarray | None = None,
    benign_label: str = "BENIGN",
) -> dict[str, Any]:
    """Add per-attack recall where original attack labels are available."""
    metrics = calculate_metrics(y_true, y_pred)
    if attack_labels is None:
        return metrics

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    labels = np.asarray(attack_labels, dtype=str).reshape(-1)
    if labels.shape != truth.shape:
        raise ValueError("attack_labels must have the same number of samples as y_true")

    benign_key = str(benign_label).strip().casefold()
    normalized_labels = np.asarray([str(label).strip() for label in labels], dtype=str)
    canonical_by_key: dict[str, str] = {}
    for label in normalized_labels:
        key = label.casefold()
        if key == benign_key:
            continue
        canonical_by_key.setdefault(key, label)

    per_attack: dict[str, float] = {}
    for key, label in sorted(canonical_by_key.items(), key=lambda item: item[1].casefold()):
        mask = np.asarray(
            [str(value).strip().casefold() == key for value in normalized_labels],
            dtype=bool,
        )
        attack_count = int(mask.sum())
        per_attack[label] = (
            float(np.count_nonzero(prediction[mask] == 1) / attack_count)
            if attack_count
            else 0.0
        )
    metrics["attack_type_recall"] = per_attack
    return metrics
