"""Tests for intrusion-detection metric edge cases."""

import numpy as np

from backend.evaluate import calculate_metrics, evaluate_predictions


def test_fpr_is_zero_for_all_benign_truth_without_false_positives() -> None:
    metrics = calculate_metrics(
        np.asarray([0, 0, 0]),
        np.asarray([0, 0, 0]),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 0}


def test_fpr_is_one_when_every_benign_row_is_a_false_positive() -> None:
    metrics = calculate_metrics(
        np.asarray([0, 0]),
        np.asarray([1, 1]),
    )

    assert metrics["false_positive_rate"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 0, "tp": 0}


def test_attack_recall_preserves_original_attack_labels() -> None:
    metrics = evaluate_predictions(
        np.asarray([0, 1, 1, 1]),
        np.asarray([0, 1, 0, 1]),
        np.asarray(["BENIGN", "DDoS", "PortScan", "DDoS"]),
    )

    assert metrics["attack_type_recall"] == {"DDoS": 1.0, "PortScan": 0.0}


def test_attack_recall_uses_configured_benign_label() -> None:
    metrics = evaluate_predictions(
        np.asarray([0, 1, 1]),
        np.asarray([0, 1, 0]),
        np.asarray(["Normal Traffic", "DoS", "DDoS"]),
        benign_label="Normal Traffic",
    )

    assert metrics["attack_type_recall"] == {"DDoS": 0.0, "DoS": 1.0}


def test_attack_recall_treats_case_and_whitespace_sensitive_labels_as_same() -> None:
    metrics = evaluate_predictions(
        np.asarray([0, 1, 1, 1]),
        np.asarray([0, 1, 0, 1]),
        np.asarray([" normal traffic ", "DoS", "dos", "DDoS"]),
        benign_label="Normal Traffic",
    )

    assert metrics["attack_type_recall"] == {"DDoS": 1.0, "DoS": 0.5}


def test_metric_functions_reject_mismatched_labels() -> None:
    try:
        calculate_metrics(np.asarray([0, 1]), np.asarray([0]))
    except ValueError as error:
        assert "same number" in str(error)
    else:
        raise AssertionError("Mismatched input sizes must raise ValueError")
