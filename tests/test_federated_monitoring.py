import numpy as np
import pytest
from fastapi import HTTPException
import yaml

from backend import api
from backend.federated import _TrafficClient


MODEL_CONFIG = {
    "type": "logistic_regression",
    "hyperparameters": {"C": 1.0, "eta0": 0.01, "learning_rate": "constant"},
}


@pytest.fixture
def editable_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {},
                "dataset": {"sample_fraction": 0.1, "source_file": "traffic.csv"},
                "preprocessing": {
                    "drop_nan_rows": True,
                    "drop_duplicate_rows": True,
                    "drop_constant_columns": True,
                },
                "split": {
                    "initial_train_fraction": 0.7,
                    "update_fraction": 0.1,
                    "test_fraction": 0.2,
                },
                "training": {
                    "random_seed": 42,
                    "clients": 3,
                    "initial_rounds": 5,
                    "update_rounds": 2,
                    "local_epochs": 1,
                    "partition": "iid",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CONFIG_PATH", config_path)
    return config_path


def test_client_reports_stable_identity_training_and_evaluation_metrics():
    client = _TrafficClient(
        np.asarray([[0.0], [1.0], [0.2], [0.8]]),
        np.asarray([0, 1, 0, 1]),
        MODEL_CONFIG,
        random_seed=42,
        local_epochs=1,
        client_id="client-2",
        partition_id="partition-2",
    )
    parameters = client.get_parameters({})

    _, samples, fit_metrics = client.fit(parameters, {})
    _, evaluated_samples, evaluation_metrics = client.evaluate(parameters, {})

    assert fit_metrics["client_id"] == "client-2"
    assert fit_metrics["partition_id"] == "partition-2"
    assert samples == evaluated_samples == 4
    assert fit_metrics["local_training_loss"] >= 0
    assert fit_metrics["training_duration_seconds"] >= 0
    assert evaluation_metrics["client_id"] == "client-2"
    assert 0 <= evaluation_metrics["f1_score"] <= 1
    assert evaluation_metrics["evaluation_duration_seconds"] >= 0


def test_client_participation_configuration_is_validated(editable_config):
    del editable_config
    payload = {
        "dataset": {"sample_fraction": 0.25},
        "preprocessing": {
            "drop_nan_rows": False,
            "drop_duplicate_rows": True,
            "drop_constant_columns": True,
        },
        "split": {
            "initial_train_fraction": 0.6,
            "update_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "training": {
            "random_seed": 13,
            "clients": 3,
            "initial_rounds": 8,
            "update_rounds": 3,
            "local_epochs": 2,
            "partition": "iid",
            "participation_fraction": 0.67,
            "min_participating_clients": 2,
        },
    }

    saved = api.update_config(payload)

    assert saved["training"]["participation_fraction"] == 0.67
    assert saved["training"]["min_participating_clients"] == 2


def test_client_participation_configuration_rejects_impossible_minimum(editable_config):
    del editable_config
    payload = {
        "dataset": {"sample_fraction": 0.25},
        "preprocessing": {
            "drop_nan_rows": False,
            "drop_duplicate_rows": True,
            "drop_constant_columns": True,
        },
        "split": {
            "initial_train_fraction": 0.6,
            "update_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "training": {
            "random_seed": 13,
            "clients": 3,
            "initial_rounds": 8,
            "update_rounds": 3,
            "local_epochs": 2,
            "partition": "iid",
            "participation_fraction": 0.5,
            "min_participating_clients": 4,
        },
    }

    with pytest.raises(HTTPException, match="cannot exceed"):
        api.update_config(payload)
