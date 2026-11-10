import json

import pandas as pd

from research.analyze import analyze_dataset, summarize_experiment


def test_analyze_dataset_reports_chunked_quality_summary(tmp_path):
    dataset = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "Attack Type": ["Normal Traffic", "DDoS", "DDoS"],
            "bytes": [1.0, 2.0, 2.0],
        }
    ).to_csv(dataset, index=False)

    summary = analyze_dataset(dataset, chunksize=2)

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["features"] == 1
    assert summary["duplicate_rows"] == 1
    assert summary["class_distribution"] == {"DDoS": 2, "Normal Traffic": 1}


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
