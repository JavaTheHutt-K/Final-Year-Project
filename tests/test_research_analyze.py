import json

import pandas as pd
import pytest

from research.analyze import analyze_dataset, summarize_experiment


def write_dataset(tmp_path, rows):
    dataset = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    return dataset


def test_analyze_dataset_reports_chunked_quality_summary(tmp_path):
    dataset = write_dataset(
        tmp_path,
        {
            "Attack Type": ["Normal Traffic", "DDoS", "DDoS"],
            "bytes": [1.0, 2.0, 2.0],
        },
    )
    summary = analyze_dataset(dataset, chunksize=2)

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["features"] == 1
    assert summary["duplicate_rows"] == 1
    assert summary["class_distribution"] == {"DDoS": 2, "Normal Traffic": 1}


@pytest.mark.parametrize(
    ("rows", "chunksize", "expected_duplicates"),
    [
        (
            {"Attack Type": ["DDoS", "DDoS"], "bytes": [2, 2]},
            10,
            1,
        ),
        (
            {"Attack Type": ["DDoS", "Normal", "DDoS"], "bytes": [2, 3, 2]},
            2,
            1,
        ),
        (
            {"Attack Type": ["DDoS", "DDoS", "DDoS"], "bytes": [2, 2, 2]},
            2,
            2,
        ),
        (
            {"Attack Type": ["DDoS", "Normal"], "bytes": [2, 3]},
            1,
            0,
        ),
    ],
)
def test_duplicate_detection_across_and_within_chunks(
    tmp_path, rows, chunksize, expected_duplicates
):
    dataset = write_dataset(tmp_path, rows)

    assert analyze_dataset(dataset, chunksize=chunksize)["duplicate_rows"] == (
        expected_duplicates
    )


def test_numeric_statistics_are_stable_for_mixed_values_and_missing_values(tmp_path):
    dataset = write_dataset(
        tmp_path,
        {
            "Attack Type": ["DDoS", "Normal", "DDoS", "Normal"],
            "bytes": ["1", "not-a-number", "", "3"],
            "packets": ["2", "", "4", "6"],
        },
    )

    summary = analyze_dataset(dataset, chunksize=2)

    assert summary["missing_values"]["bytes"] == 1
    assert summary["missing_values"]["packets"] == 1
    assert summary["numeric_summary"]["bytes"]["mean"] == 2.0
    assert summary["numeric_summary"]["packets"]["mean"] == 4.0
    assert summary["features"] == 2


def test_analysis_validation_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_dataset(tmp_path / "missing.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text("Attack Type,bytes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one row"):
        analyze_dataset(empty)

    dataset = write_dataset(tmp_path, {"Attack Type": ["DDoS"], "bytes": [1]})
    with pytest.raises(ValueError, match="chunksize"):
        analyze_dataset(dataset, chunksize=0)
    with pytest.raises(ValueError, match="label column"):
        analyze_dataset(dataset, label_column="missing")


def test_duplicate_detection_can_be_disabled(tmp_path):
    dataset = write_dataset(
        tmp_path,
        {"Attack Type": ["DDoS", "DDoS"], "bytes": [1, 1]},
    )

    summary = analyze_dataset(dataset, duplicate_detection="none")

    assert summary["duplicate_rows"] == 0


def test_summarize_experiment_preserves_measured_metrics(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "metrics": {"centralised": {"test": {"f1_score": 0.8}}},
                "training_history": [{"round": 1}],
                "evaluation": {"test_rows": 10},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment(results)

    assert summary["models"]["centralised"]["test"]["f1_score"] == 0.8
    assert summary["training_history_rounds"] == 1
    assert summary["evaluation"]["test_rows"] == 10


def test_summarize_experiment_requires_metrics_object(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"evaluation": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics object"):
        summarize_experiment(results)


def test_summarize_experiment_rejects_invalid_json(tmp_path):
    results = tmp_path / "results.json"
    results.write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        summarize_experiment(results)
