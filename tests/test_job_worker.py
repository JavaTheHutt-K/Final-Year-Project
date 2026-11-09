import numpy as np
import pytest

from backend import job_worker


def test_worker_persists_progress_history_and_completed_results(tmp_path, monkeypatch):
    run_id = "00000000-0000-4000-8000-000000000001"
    run = {"id": run_id, "status": "queued", "training_history": []}
    updates = []
    monkeypatch.setenv("IDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IDS_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(job_worker.database, "get_experiment_run", lambda _run_id: run)
    monkeypatch.setattr(job_worker.database, "get_setting", lambda _key: {"dataset": {}, "training": {}, "paths": {}})
    monkeypatch.setattr(job_worker.database, "update_experiment_run", lambda _run_id, data: updates.append(data))
    monkeypatch.setattr(job_worker, "_download_dataset", lambda _config, _path: tmp_path / "data" / "dataset.csv")

    def fake_run_experiment(config_path, progress_callback):
        progress_callback("federated_training", {"round": 1, "model": "federated"})
        return {"training_history": [{"round": 1}], "metrics": {"score": np.float64(0.8)}}

    monkeypatch.setattr(job_worker, "run_experiment", fake_run_experiment)
    results = job_worker.execute_run(run_id)

    assert results["metrics"]["score"] == 0.8
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["training_history"] == [{"round": 1}]
    assert updates[-1]["results"]["metrics"]["score"] == 0.8


def test_worker_marks_training_failure(tmp_path, monkeypatch):
    run_id = "00000000-0000-4000-8000-000000000002"
    updates = []
    monkeypatch.setenv("IDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IDS_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(
        job_worker.database,
        "get_experiment_run",
        lambda _run_id: {"id": run_id, "status": "running", "training_history": []},
    )
    monkeypatch.setattr(job_worker.database, "get_setting", lambda _key: {"dataset": {}, "training": {}, "paths": {}})
    monkeypatch.setattr(job_worker.database, "update_experiment_run", lambda _run_id, data: updates.append(data))
    monkeypatch.setattr(job_worker, "_download_dataset", lambda _config, _path: tmp_path / "data" / "dataset.csv")
    monkeypatch.setattr(job_worker, "run_experiment", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad dataset")))

    with pytest.raises(ValueError, match="bad dataset"):
        job_worker.execute_run(run_id)

    assert updates[-1]["status"] == "failed"
    assert "bad dataset" in updates[-1]["error"]
    assert "finished_at" in updates[-1]


def test_worker_requires_run_id(monkeypatch):
    monkeypatch.delenv("IDS_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="IDS_RUN_ID is required"):
        job_worker.main()


def test_worker_rejects_invalid_run_id():
    with pytest.raises(ValueError, match="valid UUID"):
        job_worker.execute_run("not-a-uuid")
