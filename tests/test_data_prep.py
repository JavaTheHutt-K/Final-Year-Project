"""Tests for chunked cleaning of the combined lightweight dataset."""

import pandas as pd
import yaml

from backend.data_prep import prepare_data


def test_prepare_data_cleans_labels_in_chunks_and_reports_counts(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "sample.csv"
    pd.DataFrame(
        {
            " Flow Size ": [1.0, 2.0, float("inf"), 2.0, 3.0],
            " Attack Type ": [
                "Normal Traffic",
                "DoS",
                "DDoS",
                "DoS",
                "Normal Traffic",
            ],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "paths": {
            "raw_data_dir": "data/raw",
            "data_report": "results/data_report.json",
        },
        "dataset": {
            "source_file": "sample.csv",
            "chunk_size": 2,
            "label_column": "Attack Type",
            "benign_label": "Normal Traffic",
            "attack_label_column": "attack_label",
            "binary_label_column": "binary_label",
            "sample_fraction": 1.0,
        },
        "preprocessing": {
            "drop_nan_rows": True,
            "drop_duplicate_rows": True,
        },
        "training": {"random_seed": 42},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    data, report = prepare_data(config_path)

    assert len(data) == 3
    assert data["binary_label"].tolist().count(0) == 2
    assert data["binary_label"].tolist().count(1) == 1
    assert report["sources"]["sample.csv"]["rows"]["before"] == 5
    assert report["sources"]["sample.csv"]["rows"]["after_nan_rows"] == 4
    assert report["sources"]["sample.csv"]["rows"]["after_duplicate_rows"] == 3
    assert (tmp_path / "results" / "data_report.json").is_file()
