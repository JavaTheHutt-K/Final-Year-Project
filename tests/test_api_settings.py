import yaml
import pytest
from fastapi import HTTPException

from backend import api


@pytest.fixture
def editable_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config = {
        "project": {"title": "Keep this setting"},
        "dataset": {"sample_fraction": 0.1, "source_file": "traffic.csv"},
        "preprocessing": {
            "drop_nan_rows": True,
            "drop_duplicate_rows": True,
            "drop_constant_columns": True,
            "scaler": "standard",
        },
        "split": {
            "initial_train_fraction": 0.7,
            "update_fraction": 0.1,
            "test_fraction": 0.2,
            "test_set_name": "Random holdout",
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
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(api, "CONFIG_PATH", config_path)
    monkeypatch.setattr(api, "_run_future", None)
    return config_path, config


def valid_settings():
    return {
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
            "clients": 4,
            "initial_rounds": 8,
            "update_rounds": 3,
            "local_epochs": 2,
            "partition": "class_skewed",
        },
    }


def test_update_config_saves_editable_fields_and_preserves_others(editable_config):
    config_path, original = editable_config

    updated = api.update_config(valid_settings())
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert updated["dataset"]["sample_fraction"] == 0.25
    assert updated["dataset"]["source_file"] == original["dataset"]["source_file"]
    assert updated["project"] == original["project"]
    assert saved == updated
    assert not config_path.with_suffix(".yaml.tmp").exists()


def test_update_config_rejects_split_that_does_not_total_one(editable_config):
    del editable_config
    payload = valid_settings()
    payload["split"]["test_fraction"] = 0.3

    with pytest.raises(HTTPException) as error:
        api.update_config(payload)

    assert error.value.status_code == 422
