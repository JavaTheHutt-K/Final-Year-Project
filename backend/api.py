"""FastAPI service exposing experiment results and execution controls."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from math import isclose
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.data_prep import _resolve_output_path
from backend.run_experiment import run_experiment
from backend.settings import DatabaseSettings

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("IDS_CONFIG_PATH", PROJECT_ROOT / "config.yaml"))
_executor: ThreadPoolExecutor | None = None
_state_lock = Lock()


_run_status: dict[str, Any] = {
    "status": "idle",
    "stage": "idle",
    "message": "Ready to run the experiment.",
}
_live_training_history: list[dict[str, Any]] = []
_run_future: Future[dict[str, Any]] | None = None
_ai_comparison_active = False


class AIComparisonRequest(BaseModel):
    test_set: str = Field(min_length=1, max_length=120)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the run worker with the ASGI app and release it on shutdown."""
    global _executor
    with _state_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="intrusion-experiment",
            )
    logger.info("Adaptive Intrusion Detection API started")
    try:
        yield
    finally:
        with _state_lock:
            executor = _executor
            _executor = None
            active_run = _run_future is not None and not _run_future.done()
            if active_run and _run_future is not None:
                _run_future.cancel()
        if active_run:
            logger.warning(
                "An experiment was active during shutdown and may be interrupted"
            )
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Adaptive Intrusion Detection API stopped")


app = FastAPI(
    title="Adaptive Intrusion Detection API",
    description="Experiment configuration, Flower training progress, and evaluation results.",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.state.database_settings = DatabaseSettings.from_environment()

with CONFIG_PATH.open("r", encoding="utf-8") as _config_file:
    _api_config = yaml.safe_load(_config_file)


def _cors_origins() -> list[str]:
    configured_origins = (
        os.environ.get("FRONTEND_URL", "").strip()
        or os.environ.get("IDS_CORS_ORIGINS", "")
    )
    if configured_origins.strip():
        origins = [
            origin.strip().rstrip("/")
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
    else:
        origins = _api_config["api"]["cors_origins"]

    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "CORS origins must be exact http(s) origins without paths or credentials"
            )
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def prevent_api_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    """Provide a lightweight health check for the hosting platform."""
    return {"status": "ok"}


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
            detail="Could not read the requested result file.",
        ) from error
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Expected a JSON object in {path.name}",
        )
    return data


def _results_path() -> Path:
    return _resolve_output_path(PROJECT_ROOT, _api_config["paths"]["results_file"])


def _report_path() -> Path:
    return _resolve_output_path(PROJECT_ROOT, _api_config["paths"]["data_report"])


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
                    "error": "Experiment failed. Check service logs for details.",
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


@app.put("/api/config")
def update_config(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate and save settings that can be changed through the dashboard."""
    editable_fields = {
        "dataset": {"sample_fraction"},
        "preprocessing": {
            "drop_nan_rows",
            "drop_duplicate_rows",
            "drop_constant_columns",
        },
        "split": {
            "initial_train_fraction",
            "update_fraction",
            "test_fraction",
        },
        "training": {
            "random_seed",
            "clients",
            "initial_rounds",
            "update_rounds",
            "local_epochs",
            "partition",
        },
    }
    if set(payload) != set(editable_fields):
        raise HTTPException(status_code=422, detail="Invalid settings sections")
    for section, allowed_fields in editable_fields.items():
        values = payload[section]
        if not isinstance(values, dict) or set(values) != allowed_fields:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid fields in {section} settings",
            )

    dataset = payload["dataset"]
    sample_fraction = dataset["sample_fraction"]
    if (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or not 0.01 <= sample_fraction <= 1
    ):
        raise HTTPException(
            status_code=422,
            detail="Dataset sample fraction must be between 0.01 and 1",
        )

    preprocessing = payload["preprocessing"]
    if any(type(value) is not bool for value in preprocessing.values()):
        raise HTTPException(
            status_code=422,
            detail="Preprocessing options must be true or false",
        )

    split = payload["split"]
    fractions = (
        split["initial_train_fraction"],
        split["update_fraction"],
        split["test_fraction"],
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value < 1
        for value in fractions
    ) or not isclose(sum(fractions), 1.0, abs_tol=0.001):
        raise HTTPException(
            status_code=422,
            detail="Data split fractions must be between 0 and 1 and sum to 1",
        )

    training = payload["training"]
    numeric_limits = {
        "random_seed": (0, 4_294_967_295),
        "clients": (1, 20),
        "initial_rounds": (1, 100),
        "update_rounds": (1, 100),
        "local_epochs": (1, 20),
    }
    for key, (minimum, maximum) in numeric_limits.items():
        value = training[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise HTTPException(
                status_code=422,
                detail=f"Training {key.replace('_', ' ')} must be between {minimum} and {maximum}",
            )
    partition_name = str(training["partition"]).strip().lower()
    if not isinstance(training["partition"], str) or partition_name not in {
        "iid",
        "class_skewed",
    }:
        raise HTTPException(
            status_code=422,
            detail="Training partition must be iid or class_skewed",
        )
    training["partition"] = partition_name

    with _state_lock:
        if _run_future is not None and not _run_future.done():
            raise HTTPException(
                status_code=409,
                detail="Settings cannot be changed while an experiment is running",
            )
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            for section, values in payload.items():
                config[section].update(values)
            temporary_path = CONFIG_PATH.with_suffix(".yaml.tmp")
            with temporary_path.open("w", encoding="utf-8", newline="\n") as config_file:
                yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)
            temporary_path.replace(CONFIG_PATH)
        except (OSError, yaml.YAMLError) as error:
            logger.exception("Could not save dashboard settings")
            raise HTTPException(
                status_code=500,
                detail="Could not save settings.",
            ) from error
    return config


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
def get_metrics(
    model: str,
    test_set: str,
) -> dict[str, Any]:
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


@app.post("/api/ai/compare")
def compare_with_ai(payload: AIComparisonRequest) -> dict[str, str]:
    """Request a post-run AI interpretation of saved metrics only."""
    global _ai_comparison_active
    with _state_lock:
        if _run_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="AI comparison is unavailable while an experiment is running.",
            )
        if _ai_comparison_active:
            raise HTTPException(
                status_code=409,
                detail="Another AI comparison is already in progress.",
            )
        _ai_comparison_active = True
    try:
        return _request_ai_comparison(payload)
    finally:
        with _state_lock:
            _ai_comparison_active = False


def _request_ai_comparison(payload: AIComparisonRequest) -> dict[str, str]:
    api_key = os.environ.get("AI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI comparison is not configured. Set AI_API_KEY on the backend.",
        )
    provider_url = os.environ.get(
        "AI_BASE_URL",
        "https://api.openai.com/v1/chat/completions",
    ).strip()
    parsed_provider_url = urlsplit(provider_url)
    if parsed_provider_url.scheme not in {"http", "https"} or not parsed_provider_url.netloc:
        raise HTTPException(
            status_code=500,
            detail="AI_BASE_URL must be an absolute HTTP(S) chat completions URL.",
        )

    results = _load_json(_results_path())
    if results is None:
        raise HTTPException(status_code=404, detail="No experiment results are available")

    model_names = {
        "centralised": "Centralised",
        "federated": "Federated",
        "centralised_updated": "Centralised (updated)",
        "federated_updated": "Federated (updated)",
    }
    metrics_by_model = results.get("metrics", {})
    comparison: dict[str, Any] = {}
    for model, label in model_names.items():
        metrics = metrics_by_model.get(model, {}).get(payload.test_set)
        if metrics is not None:
            comparison[label] = {
                name: metrics[name]
                for name in (
                    "precision",
                    "recall",
                    "f1_score",
                    "false_positive_rate",
                    "confusion_matrix",
                )
                if name in metrics
            }

    if not {"Centralised", "Federated"}.issubset(comparison):
        raise HTTPException(
            status_code=404,
            detail="Both centralised and federated metrics are required for this test set.",
        )

    prompt = {
        "test_set": payload.test_set,
        "metrics": comparison,
    }
    request_body = json.dumps(
        {
            "model": os.environ.get("AI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Interpret the supplied intrusion-detection evaluation "
                        "metrics. Compare centralised and federated approaches, "
                        "including updated variants if present. Explain tradeoffs "
                        "using only the supplied metrics, pay particular attention "
                        "to recall and false negatives, and do not claim statistical "
                        "significance or infer causes. State that these are measured "
                        "results and your text is an interpretation. Be concise."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = URLRequest(
        provider_url,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        logger.warning("AI comparison provider returned HTTP %s", error.code)
        raise HTTPException(
            status_code=502,
            detail="The AI comparison provider rejected the request.",
        ) from error
    except (
        URLError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        logger.warning("Could not complete AI comparison request: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Could not get an AI comparison. Please try again later.",
        ) from error

    try:
        analysis = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        logger.warning("AI comparison provider returned an invalid response")
        raise HTTPException(
            status_code=502,
            detail="The AI comparison provider returned an invalid response.",
        ) from error
    if not isinstance(analysis, str) or not analysis.strip():
        raise HTTPException(
            status_code=502,
            detail="The AI comparison provider returned an empty analysis.",
        )
    return {"test_set": payload.test_set, "analysis": analysis.strip()}


@app.post("/api/run", status_code=202)
def start_run() -> dict[str, str]:
    """Start an experiment in the background without blocking the API."""
    global _executor, _run_future, _live_training_history
    with _state_lock:
        if _ai_comparison_active:
            raise HTTPException(
                status_code=409,
                detail="An experiment cannot start during an AI comparison.",
            )
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
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="intrusion-experiment",
            )
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
        host=os.environ.get("HOST", str(_api_config["api"]["host"])),
        port=int(os.environ.get("PORT", _api_config["api"]["port"])),
    )
