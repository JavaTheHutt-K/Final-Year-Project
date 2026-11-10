"""FastAPI service exposing experiment results and execution controls."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from math import isclose, isfinite
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend import database
from backend.auth import (
    issue_admin_token,
    require_admin,
    verify_admin_password,
)
from backend.cloudrun import launch_training_job
from backend.data_prep import _resolve_output_path
from backend.settings import DatabaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _optional_env(name: str, default: str | None = None) -> str | None:
    """Return a trimmed environment value or the default when the variable is blank."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


logging.basicConfig(
    level=(_optional_env("LOG_LEVEL", "INFO") or "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


CONFIG_PATH = Path(_optional_env("IDS_CONFIG_PATH") or PROJECT_ROOT / "config.yaml")
_state_lock = Lock()
_ai_comparison_active = False


class AIComparisonRequest(BaseModel):
    test_set: str = Field(min_length=1, max_length=120)
    question: str | None = Field(default=None, max_length=2000)


class InsightsRequest(BaseModel):
    question: str | None = Field(default=None, max_length=2000)


class AdminLoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Run API lifecycle without starting any local training workers."""
    logger.info("Adaptive Intrusion Detection API started")
    yield
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
        if os.environ.get("K_SERVICE", "").strip():
            raise ValueError(
                "FRONTEND_URL or IDS_CORS_ORIGINS is required on Cloud Run deployments"
            )
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
    allow_headers=["Authorization", "Content-Type"],
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


def _database_is_configured() -> bool:
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_KEY", "").strip()
    )


def _database_is_required() -> bool:
    """Require durable storage on managed deployments, but allow local development."""
    return (
        os.environ.get("IDS_REQUIRE_DATABASE", "").strip().lower() in {"1", "true", "yes"}
        or bool(os.environ.get("K_SERVICE", "").strip())
    )


def _require_database_configuration() -> None:
    if _database_is_required() and not _database_is_configured():
        raise HTTPException(
            status_code=503,
            detail="Experiment storage is not configured for this deployment.",
        )


def _db_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except database.DatabaseError as error:
        logger.error("Supabase operation failed: %s", error)
        raise HTTPException(
            status_code=503,
            detail=f"Experiment storage is unavailable: {error}",
        ) from error


def _decode_config(value: Any) -> dict[str, Any]:
    config = json.loads(value) if isinstance(value, str) else value
    if not isinstance(config, dict):
        raise HTTPException(status_code=500, detail="Saved experiment configuration is invalid.")
    return config


def _configured_config() -> dict[str, Any]:
    _require_database_configuration()
    if _database_is_configured():
        saved = _db_call(database.get_setting, "experiment_config")
        if saved is not None:
            return _decode_config(saved)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as error:
        logger.exception("Could not read experiment configuration")
        raise HTTPException(status_code=500, detail="Could not read experiment configuration.") from error
    if not isinstance(config, dict):
        raise HTTPException(status_code=500, detail="Experiment configuration is invalid.")
    return config


def _write_local_config(config: dict[str, Any]) -> None:
    """Keep the local YAML copy useful for development and image inspection."""
    temporary_path = CONFIG_PATH.with_suffix(".yaml.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)
    temporary_path.replace(CONFIG_PATH)


def _validate_run_config(config: dict[str, Any]) -> None:
    dataset = config.get("dataset")
    training = config.get("training")
    split = config.get("split")
    if not isinstance(dataset, dict) or not isinstance(training, dict) or not isinstance(split, dict):
        raise HTTPException(status_code=422, detail="Experiment configuration is incomplete.")
    source_file = dataset.get("source_file")
    fraction = dataset.get("sample_fraction")
    if not isinstance(source_file, str) or not source_file.strip():
        raise HTTPException(status_code=422, detail="Dataset source_file is required.")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise HTTPException(status_code=422, detail="Dataset sample_fraction must be between 0 and 1.")
    for field in ("clients", "initial_rounds", "update_rounds", "local_epochs", "random_seed"):
        value = training.get(field)
        if type(value) is not int or value < 0 or (field != "random_seed" and value == 0):
            raise HTTPException(status_code=422, detail=f"Training {field} is invalid.")
    participation_fraction = training.get("participation_fraction", 1.0)
    if (
        isinstance(participation_fraction, bool)
        or not isinstance(participation_fraction, (int, float))
        or not isfinite(participation_fraction)
        or not 0 < participation_fraction <= 1
    ):
        raise HTTPException(status_code=422, detail="Training participation fraction is invalid.")
    minimum_clients = training.get("min_participating_clients", training.get("clients"))
    if (
        type(minimum_clients) is not int
        or minimum_clients < 1
        or minimum_clients > training["clients"]
    ):
        raise HTTPException(status_code=422, detail="Minimum participating clients is invalid.")
    fractions = (
        split.get("initial_train_fraction"),
        split.get("update_fraction"),
        split.get("test_fraction"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or not 0 < value < 1
        for value in fractions
    ) or not isclose(sum(fractions), 1.0, abs_tol=0.001):
        raise HTTPException(status_code=422, detail="Data split fractions must be valid and sum to 1.")


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Return the saved experiment configuration or the initial YAML default."""
    return _configured_config()


@app.put("/api/config")
def update_config(
    payload: dict[str, Any],
    _: None = Depends(require_admin),
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
            "participation_fraction",
            "min_participating_clients",
        },
    }
    if set(payload) != set(editable_fields):
        raise HTTPException(status_code=422, detail="Invalid settings sections")
    required_fields = {
        section: fields - {
            "participation_fraction",
            "min_participating_clients",
        }
        for section, fields in editable_fields.items()
    }
    for section, allowed_fields in editable_fields.items():
        values = payload[section]
        if (
            not isinstance(values, dict)
            or not required_fields[section].issubset(values)
            or not set(values).issubset(allowed_fields)
        ):
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
    training.setdefault("participation_fraction", 1.0)
    training.setdefault("min_participating_clients", training["clients"])
    numeric_limits = {
        "random_seed": (0, 4_294_967_295),
        "clients": (1, 20),
        "initial_rounds": (1, 100),
        "update_rounds": (1, 100),
        "local_epochs": (1, 20),
        "min_participating_clients": (1, 20),
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
    participation_fraction = training["participation_fraction"]
    if (
        isinstance(participation_fraction, bool)
        or not isinstance(participation_fraction, (int, float))
        or not 0 < participation_fraction <= 1
    ):
        raise HTTPException(
            status_code=422,
            detail="Training participation fraction must be between 0 and 1",
        )
    if training["min_participating_clients"] > training["clients"]:
        raise HTTPException(
            status_code=422,
            detail="Minimum participating clients cannot exceed the client count",
        )

    with _state_lock:
        if _database_is_configured():
            active_run = _db_call(database.get_latest_active_run)
            if active_run:
                raise HTTPException(
                    status_code=409,
                    detail="Settings cannot be changed while an experiment is active.",
                )
        config = _configured_config()
        for section, values in payload.items():
            if not isinstance(config.get(section), dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"Stored configuration section {section!r} is invalid.",
                )
            config[section].update(values)
        if _database_is_configured():
            _db_call(database.save_setting, "experiment_config", json.dumps(config))
        try:
            _write_local_config(config)
        except (OSError, yaml.YAMLError) as error:
            if not _database_is_configured():
                logger.exception("Could not save dashboard settings")
                raise HTTPException(status_code=500, detail="Could not save settings.") from error
            logger.warning(
                "Could not update local configuration copy; Supabase remains authoritative."
            )
    return config


@app.get("/api/results")
def get_results() -> dict[str, Any]:
    """Return results from the latest successful run, never a failed run."""
    _require_database_configuration()
    if _database_is_configured():
        run = _db_call(database.get_latest_completed_run)
        if run is not None:
            results = run.get("results")
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except json.JSONDecodeError as error:
                    raise HTTPException(
                        status_code=500, detail="Saved experiment results are invalid."
                    ) from error
            if isinstance(results, dict):
                return results
            raise HTTPException(
                status_code=500,
                detail="The latest completed experiment has no valid saved results.",
            )
        return _empty_results()

    results = _load_json(_results_path())
    if results is not None:
        return results
    return _empty_results()


def _empty_results() -> dict[str, Any]:
    response: dict[str, Any] = {
        "config": _configured_config(),
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
    _require_database_configuration()
    if _database_is_configured():
        active = _db_call(database.get_latest_active_run)
        run = active or _db_call(database.get_latest_completed_run)
        if run is None:
            return []
        history = run.get("training_history")
        if isinstance(history, str):
            try:
                history = json.loads(history)
            except json.JSONDecodeError as error:
                raise HTTPException(status_code=500, detail="Saved training history is invalid.") from error
        if not history and isinstance(run.get("results"), dict):
            saved_history = run["results"].get("training_history", [])
            if isinstance(saved_history, list):
                history = saved_history
        if not isinstance(history, list):
            raise HTTPException(status_code=500, detail="Saved training history is invalid.")
        return history
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
    results = get_results()
    if not results.get("metrics"):
        raise HTTPException(status_code=404, detail="No experiment results are available")
    metrics = results.get("metrics", {}).get(model, {}).get(test_set)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics are available for model {model!r} on test set {test_set!r}",
        )
    return {"model": model, "test_set": test_set, **metrics}


@app.post("/api/ai/compare")
def compare_with_ai(
    payload: AIComparisonRequest,
    _: None = Depends(require_admin),
) -> dict[str, str]:
    """Request a post-run AI interpretation of saved metrics only."""
    global _ai_comparison_active
    with _state_lock:
        if _database_is_configured() and _db_call(database.get_latest_active_run):
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


def _metric_comparison(
    results: dict[str, Any],
    test_set: str,
) -> dict[str, Any]:
    if not results.get("metrics"):
        raise HTTPException(status_code=404, detail="No completed experiment results are available.")

    labels = {
        "centralised": "Centralised",
        "federated": "Federated",
        "centralised_updated": "Centralised (updated)",
        "federated_updated": "Federated (updated)",
    }
    comparison: dict[str, Any] = {}
    metrics_by_model = results.get("metrics", {})
    for model, label in labels.items():
        metrics = metrics_by_model.get(model, {}).get(test_set)
        if isinstance(metrics, dict):
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
            detail=f"Both centralised and federated metrics are required for test set {test_set!r}.",
        )
    return comparison


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) and isfinite(value) else None


def _clean_ai_output(analysis: str) -> str:
    """Remove accidental Markdown emphasis and code delimiters from AI output."""
    cleaned_lines = []
    for line in analysis.replace("*", "").replace("`", "").splitlines():
        line = line.lstrip("#").strip()
        if "|" in line:
            line = " - ".join(part.strip() for part in line.split("|") if part.strip())
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _saved_test_set(results: dict[str, Any]) -> str | None:
    metrics_by_model = results.get("metrics", {})
    if not isinstance(metrics_by_model, dict):
        return None
    for model in ("centralised", "federated", "centralised_updated", "federated_updated"):
        model_metrics = metrics_by_model.get(model)
        if isinstance(model_metrics, dict) and model_metrics:
            return next(iter(model_metrics))
    return None


def _deterministic_comparison(test_set: str, comparison: dict[str, Any]) -> str:
    """Create a transparent interpretation without an external AI service."""
    lines = [
        f"1. Experimental Overview\nDeterministic fallback analysis for the measured {test_set} results.",
        "This interpretation uses only metrics saved by the experiment; it is not Groq-generated.",
    ]
    for label, metrics in comparison.items():
        measured = "; ".join(
            f"{name.replace('_', ' ')} {value:.4f}"
            for name in ("precision", "recall", "f1_score", "false_positive_rate")
            if (value := _metric_value(metrics, name)) is not None
        )
        confusion_matrix = metrics.get("confusion_matrix")
        false_negatives = (
            confusion_matrix.get("fn")
            if isinstance(confusion_matrix, dict)
            and isinstance(confusion_matrix.get("fn"), (int, float))
            else None
        )
        if false_negatives is not None:
            measured += f"; false negatives {false_negatives:.0f}"
        if measured:
            lines.append(f"{label}: {measured}.")

    leaders = (
        ("highest F1-score", "f1_score", max),
        ("highest recall", "recall", max),
        ("lowest false-positive rate", "false_positive_rate", min),
    )
    for description, metric_name, selector in leaders:
        available = [
            (label, value)
            for label, metrics in comparison.items()
            if (value := _metric_value(metrics, metric_name)) is not None
        ]
        if available:
            best_value = selector(value for _, value in available)
            best_labels = [label for label, value in available if value == best_value]
            lines.append(
                f"The {description} is {best_value:.4f}, recorded by "
                f"{', '.join(best_labels)}."
            )
    false_negative_counts = [
        (label, metrics["confusion_matrix"]["fn"])
        for label, metrics in comparison.items()
        if isinstance(metrics.get("confusion_matrix"), dict)
        and isinstance(metrics["confusion_matrix"].get("fn"), (int, float))
    ]
    if false_negative_counts:
        best_fn = min(value for _, value in false_negative_counts)
        best_labels = [label for label, value in false_negative_counts if value == best_fn]
        lines.append(
            f"The lowest number of false negatives is {best_fn:.0f}, recorded by "
            f"{', '.join(best_labels)}."
        )
    lines.append(
        "Recall indicates the proportion of malicious samples detected; false negatives "
        "are attacks that were not detected. Precision reflects the proportion of "
        "predicted attacks that were correct, while the false-positive rate reflects "
        "benign samples incorrectly flagged."
    )
    lines.append("These comparisons describe this holdout only and do not establish statistical significance.")
    return "\n".join(lines)


def _request_ai_comparison(payload: AIComparisonRequest) -> dict[str, Any]:
    results = get_results()
    comparison = _metric_comparison(results, payload.test_set)
    fallback = _deterministic_comparison(payload.test_set, comparison)
    provider = (_optional_env("AI_PROVIDER", "groq") or "groq").lower()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = _optional_env("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    fallback_response = {
        "test_set": payload.test_set,
        "analysis": fallback,
        "source": "deterministic-fallback",
        "model": "deterministic-fallback",
    }
    if provider != "groq":
        logger.warning("Unsupported AI_PROVIDER=%s; returning deterministic comparison.", provider)
        return fallback_response
    if not api_key:
        logger.info("Groq is not configured; returning deterministic comparison.")
        return fallback_response

    question = payload.question.strip() if payload.question and payload.question.strip() else None
    if question:
        prompt = (
            "Answer the following question directly using only the supplied saved "
            "intrusion-detection metrics. Begin with the direct answer, then include "
            "only relevant sections from: 1. Direct Answer, 2. Relevant Experimental "
            "Evidence, 3. Technical Interpretation, 4. Comparison of Model Performance, "
            "5. Limitations, and 6. Conclusion. Reference measured metrics, explain "
            "precision, recall, F1-score, false-positive rate, false positives, and "
            "false negatives accurately, and explain the practical meaning of missed "
            "attacks and false alarms when relevant. Compare centralised and federated "
            "approaches when relevant. If the supplied data is insufficient, state "
            "that explicitly. Avoid repetition and unsupported explanations. State "
            "that conclusions apply only to the supplied holdout test set. Do not "
            "invent causes, statistical significance, or findings. Format metrics to "
            "four decimal places where appropriate.\n\n"
            f"Question: {question}\n\n"
        )
    else:
        prompt = (
            "Prepare a professional experimental analysis of the supplied intrusion-detection "
            "evaluation metrics for a final-year Computer Science research project. Begin "
            "with a brief explanation of what is being compared. Use the following numbered "
            "sections: 1. Experimental Overview, 2. Original Centralised vs Federated "
            "Comparison, 3. Updated Centralised vs Federated Comparison, 4. Precision "
            "and False-Positive Analysis, 5. Recall and False-Negative Analysis, "
            "6. Overall F1-Score Comparison, 7. Performance Trade-offs, 8. Research "
            "Findings and Conclusion, and 9. Experimental Limitations. Omit sections "
            "that are not relevant or cannot be supported by the supplied metrics. "
            "Compare all available configurations and identify the highest F1-score, "
            "highest recall, lowest false-positive rate, and lowest number of false "
            "negatives. Explain why these metrics matter in intrusion detection without "
            "inventing causes or claiming universal superiority. State when a metric "
            "was not provided and that conclusions apply only to the supplied holdout "
            "test set. Use only the supplied metrics and format them to four decimal "
            "places where appropriate.\n\n"
        )
    prompt += json.dumps({"test_set": payload.test_set, "metrics": comparison})
    try:
        from groq import Groq

        response = Groq(api_key=api_key, timeout=30.0).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You interpret saved intrusion-detection metrics only. "
                        "Return plain text only. Never use asterisks, Markdown bold, "
                        "Markdown italics, Markdown tables, backticks, or Markdown "
                        "headings beginning with #, or pipe characters used as table "
                        "delimiters. Use numbered section headings appropriate to the "
                        "request. For comparisons, use sections for the experimental "
                        "overview, original and updated centralised versus federated "
                        "performance, precision and false-positive analysis, recall "
                        "and false-negative analysis, F1-score comparison, trade-offs, "
                        "findings, and limitations. For questions, use only relevant "
                        "sections for the direct answer, evidence, interpretation, "
                        "comparison, limitations, and conclusion. Use short, "
                        "well-structured paragraphs and "
                        "numbered lists where appropriate. Use professional academic "
                        "language suitable for a final-year Computer Science research "
                        "project. Format metrics consistently to four decimal places "
                        "where appropriate. Avoid repetition and unnecessary "
                        "introductory language. Clearly distinguish measured results "
                        "from interpretations. Do not invent metrics, causes, "
                        "statistical significance, or experimental findings. State "
                        "when a metric was not provided, explain recall, false "
                        "negatives, precision, false positives, and F1-score "
                        "accurately, and compare all four configurations when metrics "
                        "are available. Explicitly state that conclusions apply only "
                        "to the supplied holdout test set. Do not request or infer "
                        "dataset contents."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
    except ImportError:
        logger.warning("Groq support is unavailable; returning deterministic comparison.")
        return fallback_response
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403}:
            reason = "credentials were rejected"
        elif status_code == 404:
            reason = "the configured model is unavailable"
        elif status_code == 429:
            reason = "the request was rate-limited or the quota is exhausted"
        elif error.__class__.__name__ in {"APITimeoutError", "TimeoutError"}:
            reason = "the request timed out"
        else:
            reason = "the provider request failed"
        logger.warning("Groq comparison failed for model %s: %s; using fallback.", model, reason)
        return fallback_response
    choices = getattr(response, "choices", None)
    message = choices[0].message if isinstance(choices, list) and choices else None
    analysis = getattr(message, "content", None)
    if not isinstance(analysis, str) or not analysis.strip():
        logger.warning("Groq returned an empty comparison; returning deterministic comparison.")
        return fallback_response
    cleaned_analysis = _clean_ai_output(analysis)
    if not cleaned_analysis:
        logger.warning("Groq returned no usable comparison after output cleaning; returning deterministic comparison.")
        return fallback_response
    return {
        "test_set": payload.test_set,
        "analysis": cleaned_analysis,
        "source": "groq",
        "model": model,
    }


@app.post("/api/insights")
def generate_insights(
    payload: InsightsRequest | None = None,
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    """Generate an interpretation of the latest successful experiment metrics."""
    global _ai_comparison_active
    with _state_lock:
        if _database_is_configured() and _db_call(database.get_latest_active_run):
            raise HTTPException(status_code=409, detail="Groq analysis is unavailable while an experiment is running.")
        if _ai_comparison_active:
            raise HTTPException(status_code=409, detail="Another Groq analysis is already in progress.")
        _ai_comparison_active = True
    try:
        results = get_results()
        test_set = _saved_test_set(results)
        if not test_set:
            raise HTTPException(status_code=404, detail="No completed experiment metrics are available.")
        comparison = _request_ai_comparison(
            AIComparisonRequest(
                test_set=test_set,
                question=(
                    payload.question.strip()
                    if payload and payload.question and payload.question.strip()
                    else None
                ),
            )
        )
        latest = _db_call(database.get_latest_completed_run) if _database_is_configured() else None
        return {
            "run_id": str(latest["id"]) if latest else "local-results",
            "model": comparison["model"],
            "source": comparison["source"],
            "insights": comparison["analysis"],
        }
    finally:
        with _state_lock:
            _ai_comparison_active = False


@app.post("/api/run", status_code=202)
def start_run(_: None = Depends(require_admin)) -> dict[str, Any]:
    """Create a Supabase run and launch its dedicated Cloud Run Job."""
    return start_run_for_authorized_admin()


def start_run_for_authorized_admin() -> dict[str, Any]:
    """Shared orchestration for the authenticated API and Gradio admin action."""
    with _state_lock:
        if _ai_comparison_active:
            raise HTTPException(
                status_code=409,
                detail="An experiment cannot start during an AI comparison.",
            )
        if not _database_is_configured():
            raise HTTPException(
                status_code=503,
                detail="Experiment storage is not configured; set SUPABASE_URL and SUPABASE_KEY.",
            )
        if _db_call(database.get_latest_active_run):
            raise HTTPException(status_code=409, detail="An experiment is already queued or running.")
        config = _configured_config()
        _validate_run_config(config)
        run_id = str(uuid4())
        _db_call(database.save_setting, "experiment_config", json.dumps(config))
        _db_call(
            database.create_experiment_run,
            {
                "id": run_id,
                "status": "queued",
                "stage": "queued",
                "message": "Cloud Run Job is being launched.",
                "error": None,
                "training_history": [],
                "results": None,
            },
        )
        try:
            execution = launch_training_job(run_id)
        except Exception as error:
            logger.exception("Could not launch Cloud Run Job for run %s", run_id)
            try:
                _db_call(
                    database.update_experiment_run,
                    run_id,
                    {
                        "status": "failed",
                        "stage": "failed",
                        "message": "Cloud Run Job launch failed.",
                        "error": str(error)[:1000],
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except HTTPException:
                logger.exception("Could not mark run %s failed after launch failure", run_id)
            raise HTTPException(
                status_code=502,
                detail="The experiment run was created but the Cloud Run Job could not be launched.",
            ) from error
        _db_call(
            database.update_experiment_run,
            run_id,
            {
                "cloud_run_execution": execution,
                "message": "Cloud Run Job execution launched.",
            },
        )
    return {
        "id": run_id,
        "run_id": run_id,
        "status": "queued",
        "message": "Cloud Run Job execution launched.",
    }


@app.get("/api/run/status")
def get_run_status() -> dict[str, Any]:
    """Return the latest authoritative run state from Supabase."""
    if not _database_is_configured():
        raise HTTPException(
            status_code=503,
            detail="Experiment storage is not configured; run status is unavailable.",
        )
    run = _db_call(database.get_latest_active_run) or _db_call(database.get_latest_run)
    if run is None:
        return {
            "status": "idle",
            "stage": "idle",
            "message": "Ready to run the experiment.",
            "run_id": None,
        }
    history = run.get("training_history")
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except json.JSONDecodeError:
            history = []
    latest = history[-1] if isinstance(history, list) and history else {}
    return {
        "run_id": str(run.get("id", "")),
        "status": run.get("status", "unavailable"),
        "stage": run.get("stage"),
        "message": run.get("message"),
        "error": run.get("error"),
        "cloud_run_execution": run.get("cloud_run_execution"),
        "round": latest.get("round"),
        "expected_clients": latest.get("expected_clients"),
        "selected_clients": latest.get("selected_clients", []),
        "successful_participants": latest.get("successful_participants", []),
        "failed_participants": latest.get("failed_participants", []),
        "aggregation_completed": latest.get("aggregation_completed"),
        "client_details": latest.get("client_details", []),
    }


@app.post("/api/auth/login")
def admin_login(payload: AdminLoginRequest) -> dict[str, Any]:
    """Authenticate an administrator using the backend password hash."""
    if not os.environ.get("ADMIN_PASSWORD_HASH", "").strip():
        raise HTTPException(status_code=503, detail="Administrator login is not configured.")
    if not verify_admin_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid administrator password.")
    return {
        "access_token": issue_admin_token(),
        "token_type": "bearer",
        "expires_in": 3600,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=_optional_env("HOST") or str(_api_config["api"]["host"]),
        port=int(_optional_env("PORT") or _api_config["api"]["port"]),
    )
