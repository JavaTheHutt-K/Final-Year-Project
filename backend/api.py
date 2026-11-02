"""FastAPI service exposing experiment results and execution controls."""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.run_experiment import run_experiment

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intrusion-experiment")
_state_lock = Lock()
_run_status: dict[str, Any] = {
    "status": "idle",
    "stage": "idle",
    "message": "Ready to run the experiment.",
}
_live_training_history: list[dict[str, Any]] = []
_run_future: Future[dict[str, Any]] | None = None

app = FastAPI(
    title="Adaptive Intrusion Detection API",
    description="Experiment configuration, Flower training progress, and evaluation results.",
    version="0.1.0",
)

with CONFIG_PATH.open("r", encoding="utf-8") as _config_file:
    _api_config = yaml.safe_load(_config_file)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read an existing JSON output file, reporting corrupt files explicitly."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("Could not read backend output file %s", path)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {error}",
        ) from error
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Expected a JSON object in {path.name}",
        )
    return data


def _results_path() -> Path:
    configured = Path(_api_config["paths"]["results_file"])
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _report_path() -> Path:
    configured = Path(_api_config["paths"]["data_report"])
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _run_progress(stage: str, detail: dict[str, Any]) -> None:
    """Update observable run progress from the experiment worker."""
    with _state_lock:
        if "round" in detail and "model" in detail:
            _live_training_history.append(dict(detail))
        _run_status.update(
            {
                "status": "running",
                "stage": stage,
                "message": stage.replace("_", " ").capitalize(),
                **detail,
            }
        )


def _execute_experiment() -> dict[str, Any]:
    """Run the long experiment on a worker and preserve explicit failures."""
    global _run_future
    try:
        results = run_experiment(CONFIG_PATH, progress_callback=_run_progress)
        with _state_lock:
            _run_status.update(
                {
                    "status": "completed",
                    "stage": "completed",
                    "message": "Experiment completed successfully.",
                    "error": None,
                }
            )
        return results
    except Exception as error:
        logger.exception("Experiment run failed")
        with _state_lock:
            _run_status.update(
                {
                    "status": "failed",
                    "stage": "failed",
                    "message": "Experiment failed.",
                    "error": str(error),
                }
            )
        raise
    finally:
        with _state_lock:
            _run_future = None


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Return the active root YAML configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


@app.get("/api/results")
def get_results() -> dict[str, Any]:
    """Return the latest result payload or a typed empty initial response."""
    results = _load_json(_results_path())
    if results is not None:
        return results
    response: dict[str, Any] = {
        "config": _api_config,
        "training_history": [],
        "metrics": {
            "federated": {},
            "centralised": {},
            "federated_updated": {},
            "centralised_updated": {},
        },
    }
    report = _load_json(_report_path())
    if report is not None:
        response["data_report"] = report
    return response


@app.get("/api/training-history")
def get_training_history() -> list[dict[str, Any]]:
    """Return per-round federated training and evaluation history."""
    with _state_lock:
        if _run_status["status"] == "running":
            return [dict(entry) for entry in _live_training_history]
    results = _load_json(_results_path())
    if results is None:
        return []
    history = results.get("training_history", [])
    if not isinstance(history, list):
        raise HTTPException(
            status_code=500,
            detail="training_history in results.json must be an array",
        )
    return history


@app.get("/api/metrics")
def get_metrics(model: str, test_set: str) -> dict[str, Any]:
    """Return metrics for one model and the named holdout test set."""
    if model not in {
        "federated",
        "centralised",
        "federated_updated",
        "centralised_updated",
    }:
        raise HTTPException(status_code=422, detail="Unknown model identifier")
    results = _load_json(_results_path())
    if results is None:
        raise HTTPException(status_code=404, detail="No experiment results are available")
    metrics = results.get("metrics", {}).get(model, {}).get(test_set)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics are available for model {model!r} on test set {test_set!r}",
        )
    return {"model": model, "test_set": test_set, **metrics}


@app.post("/api/run", status_code=202)
def start_run() -> dict[str, str]:
    """Start an experiment in the background without blocking the API."""
    global _run_future, _live_training_history
    with _state_lock:
        if _run_future is not None and not _run_future.done():
            raise HTTPException(status_code=409, detail="An experiment is already running")
        for key in ("model", "round", "clients"):
            _run_status.pop(key, None)
        _run_status.update(
            {
                "status": "running",
                "stage": "starting",
                "message": "Experiment queued.",
                "error": None,
            }
        )
        _live_training_history = []
        _run_future = _executor.submit(_execute_experiment)
    return {"status": "running", "message": "Experiment started in the background."}


@app.get("/api/run/status")
def get_run_status() -> dict[str, Any]:
    """Return current run state, stage, and any explicit error."""
    with _state_lock:
        return dict(_run_status)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=str(_api_config["api"]["host"]),
        port=int(_api_config["api"]["port"]),
    )
