import pytest
from fastapi import HTTPException

from backend import api


def configure_database(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")


def test_run_creates_persists_and_launches_cloud_job(monkeypatch):
    configure_database(monkeypatch)
    calls = []
    monkeypatch.setattr(api.database, "get_latest_active_run", lambda: None)
    monkeypatch.setattr(api, "_configured_config", valid_config)
    monkeypatch.setattr(api.database, "save_setting", lambda *args: calls.append(("setting", args)))
    monkeypatch.setattr(api.database, "create_experiment_run", lambda run: calls.append(("create", run)))
    monkeypatch.setattr(api, "launch_training_job", lambda run_id: {"operation": "operations/123"})
    monkeypatch.setattr(api.database, "update_experiment_run", lambda *args: calls.append(("update", args)))

    response = api.start_run_for_authorized_admin()

    created = next(call[1] for call in calls if call[0] == "create")
    assert response["status"] == "queued"
    assert response["run_id"] == created["id"]
    assert created["status"] == "queued"
    assert any(call[0] == "update" for call in calls)


def test_run_launch_failure_marks_run_failed(monkeypatch):
    configure_database(monkeypatch)
    updates = []
    monkeypatch.setattr(api.database, "get_latest_active_run", lambda: None)
    monkeypatch.setattr(api, "_configured_config", valid_config)
    monkeypatch.setattr(api.database, "save_setting", lambda *_args: None)
    monkeypatch.setattr(api.database, "create_experiment_run", lambda _run: None)
    monkeypatch.setattr(api, "launch_training_job", lambda _run_id: (_ for _ in ()).throw(RuntimeError("job unavailable")))
    monkeypatch.setattr(api.database, "update_experiment_run", lambda *args: updates.append(args))

    with pytest.raises(HTTPException) as error:
        api.start_run_for_authorized_admin()

    assert error.value.status_code == 502
    assert updates[0][1]["status"] == "failed"
    assert "job unavailable" in updates[0][1]["error"]


def test_concurrent_run_is_rejected(monkeypatch):
    configure_database(monkeypatch)
    monkeypatch.setattr(api.database, "get_latest_active_run", lambda: {"id": "existing"})

    with pytest.raises(HTTPException) as error:
        api.start_run_for_authorized_admin()

    assert error.value.status_code == 409


def valid_config():
    return {
        "dataset": {"source_file": "traffic.csv", "sample_fraction": 0.1},
        "training": {
            "clients": 3,
            "initial_rounds": 5,
            "update_rounds": 2,
            "local_epochs": 1,
            "random_seed": 42,
        },
        "split": {
            "initial_train_fraction": 0.7,
            "update_fraction": 0.1,
            "test_fraction": 0.2,
        },
    }


def test_latest_success_results_survive_newer_failed_run(monkeypatch):
    configure_database(monkeypatch)
    successful = {"id": "success", "results": {"metrics": {"federated": {"holdout": {}}}}}
    monkeypatch.setattr(api.database, "get_latest_completed_run", lambda: successful)
    assert api.get_results() == successful["results"]
