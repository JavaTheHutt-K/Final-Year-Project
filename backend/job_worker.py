"""Cloud Run Job entry point for durable experiment execution."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from backend import database
from backend.run_experiment import run_experiment

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert NumPy and path values to JSON-native values without losing content."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Experiment result contains non-JSON value {type(value).__name__}.")


def _load_saved_config() -> dict[str, Any]:
    raw = database.get_setting("experiment_config")
    if raw is None:
        raise RuntimeError("The saved experiment_config setting was not found.")
    config = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(config, dict):
        raise RuntimeError("The saved experiment configuration is invalid.")
    return config


def _download_dataset(config: dict[str, Any], data_dir: Path) -> Path:
    from google.cloud import storage

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError("The saved experiment configuration has no valid dataset section.")
    bucket_name = os.environ.get("IDS_GCS_BUCKET", "").strip() or dataset.get("gcs_bucket")
    object_name = dataset.get("gcs_object") or dataset.get("source_file")
    if not isinstance(bucket_name, str) or not bucket_name.strip():
        raise RuntimeError("Set IDS_GCS_BUCKET or dataset.gcs_bucket for the worker.")
    if not isinstance(object_name, str) or not object_name.strip():
        raise RuntimeError("The experiment dataset object is not configured.")
    filename = Path(object_name).name
    if not filename or filename in {".", ".."}:
        raise RuntimeError("The configured GCS dataset object has an invalid filename.")
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / filename
    try:
        storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(str(target))
    except Exception as error:
        raise RuntimeError("The experiment dataset could not be downloaded from Cloud Storage.") from error
    dataset["source_file"] = filename
    config.setdefault("paths", {})["raw_data_dir"] = str(data_dir)
    return target


def execute_run(run_id: str) -> dict[str, Any]:
    """Download inputs, execute real training, and persist all run state."""
    try:
        UUID(run_id)
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError("run_id must be a valid UUID.") from error
    run = database.get_experiment_run(run_id)
    if run is None:
        raise RuntimeError(f"Experiment run {run_id!r} was not found.")
    if run.get("status") not in {"queued", "running"}:
        raise RuntimeError(f"Experiment run {run_id!r} is not queued or running.")

    data_dir = Path(os.environ.get("IDS_DATA_DIR", "/tmp/ids-data")).resolve()
    results_dir = Path(os.environ.get("IDS_RESULTS_DIR", "/tmp/ids-results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    database.update_experiment_run(
        run_id,
        {
            "status": "running",
            "stage": "preparing_data",
            "message": "Worker started.",
            "started_at": _utc_now(),
            "error": None,
        },
    )
    try:
        config = _load_saved_config()
        _download_dataset(config, data_dir)
        config_path = results_dir / f"config-{run_id}.yaml"
        config["paths"]["results_dir"] = str(results_dir)
        config["paths"]["results_file"] = str(results_dir / f"results-{run_id}.json")
        config["paths"]["data_report"] = str(results_dir / f"data-report-{run_id}.json")
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        history = run.get("training_history")
        if not isinstance(history, list):
            history = []

        def progress(stage: str, detail: dict[str, Any]) -> None:
            if "round" in detail and "model" in detail:
                history.append(
                    _json_safe({**detail, "run_id": run_id})
                )
            database.update_experiment_run(
                run_id,
                {
                    "status": "running",
                    "stage": stage,
                    "message": stage.replace("_", " ").capitalize(),
                    "training_history": _json_safe(history),
                },
            )

        results = run_experiment(config_path, progress_callback=progress)
        safe_results = _json_safe(results)
        if isinstance(safe_results.get("training_history"), list):
            safe_results["training_history"] = [
                {**entry, "run_id": run_id}
                if isinstance(entry, dict)
                else entry
                for entry in safe_results["training_history"]
            ]
        json.dumps(safe_results, allow_nan=False)
        database.update_experiment_run(
            run_id,
            {
                "status": "completed",
                "stage": "completed",
                "message": "Experiment completed successfully.",
                "error": None,
                "training_history": safe_results.get("training_history", []),
                "results": safe_results,
                "finished_at": _utc_now(),
            },
        )
        return safe_results
    except Exception as error:
        logger.exception("Cloud Run experiment %s failed", run_id)
        try:
            database.update_experiment_run(
                run_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "message": "Experiment failed.",
                    "error": str(error)[:1000],
                    "finished_at": _utc_now(),
                },
            )
        except database.DatabaseError:
            logger.exception("Could not persist failure state for run %s", run_id)
        raise


def main() -> None:
    run_id = os.environ.get("IDS_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("IDS_RUN_ID is required.")
    execute_run(run_id)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    main()
