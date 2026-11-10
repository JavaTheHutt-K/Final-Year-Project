from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.auth import issue_admin_token


def saved_results():
    return {
        "metrics": {
            "centralised": {
                "Random holdout": {
                    "precision": 0.9,
                    "recall": 0.8,
                    "f1_score": 0.85,
                    "false_positive_rate": 0.1,
                    "confusion_matrix": {"tn": 90, "fp": 10, "fn": 20, "tp": 80},
                }
            },
            "federated": {
                "Random holdout": {
                    "precision": 0.88,
                    "recall": 0.84,
                    "f1_score": 0.86,
                    "false_positive_rate": 0.12,
                    "confusion_matrix": {"tn": 88, "fp": 12, "fn": 16, "tp": 84},
                }
            },
        }
    }


def test_groq_comparison_uses_saved_completed_metrics(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(api, "get_results", saved_results)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Federated recall is higher."))]
        )

    groq = SimpleNamespace(
        Groq=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "groq", groq)

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result == {
        "test_set": "Random holdout",
        "analysis": "Federated recall is higher.",
        "source": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    assert captured["model"] == "llama-3.3-70b-versatile"
    assert "Random holdout" in captured["messages"][1]["content"]
    assert "traffic" not in captured["messages"][1]["content"].lower()
    assert "cicids2017" not in captured["messages"][1]["content"].lower()


def test_ai_comparison_is_rejected_without_admin_authorization():
    with TestClient(api.app) as client:
        response = client.post("/api/ai/compare", json={"test_set": "Random holdout"})
    assert response.status_code == 401


def test_ai_comparison_is_not_available_during_training(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN_SECRET", "test-token-secret-that-is-long-enough")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "backend-key")
    monkeypatch.setattr(api.database, "get_latest_active_run", lambda: {"id": "run-active"})
    token = issue_admin_token()

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("Groq must not run while training is active")

    monkeypatch.setattr(api, "_request_ai_comparison", unexpected_request)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 409


def test_ai_comparison_uses_deterministic_fallback_without_backend_groq_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(api, "get_results", saved_results)

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"
    assert result["model"] == "deterministic-fallback"
    assert "F1-score" in result["analysis"]
    assert "not Groq-generated" in result["analysis"]


def test_ai_comparison_falls_back_when_groq_fails(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setattr(api, "get_results", saved_results)

    def failing_client(**_kwargs):
        raise RuntimeError("quota exhausted")

    monkeypatch.setitem(
        __import__("sys").modules,
        "groq",
        SimpleNamespace(Groq=failing_client),
    )

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"
    assert "measured" in result["analysis"]


def test_ai_comparison_falls_back_for_unavailable_groq_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setenv("GROQ_MODEL", "unavailable-model")
    monkeypatch.setattr(api, "get_results", saved_results)

    class ModelUnavailableError(Exception):
        status_code = 404

    monkeypatch.setitem(
        __import__("sys").modules,
        "groq",
        SimpleNamespace(
            Groq=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: (_ for _ in ()).throw(ModelUnavailableError())
                    )
                )
            )
        ),
    )

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"
    assert result["model"] == "deterministic-fallback"


@pytest.mark.parametrize("status_code", [401, 403, 429])
def test_ai_comparison_falls_back_for_groq_http_failures(monkeypatch, status_code):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setattr(api, "get_results", saved_results)

    class GroqHTTPError(Exception):
        pass

    error = GroqHTTPError()
    error.status_code = status_code
    monkeypatch.setitem(
        __import__("sys").modules,
        "groq",
        SimpleNamespace(
            Groq=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: (_ for _ in ()).throw(error)
                    )
                )
            )
        ),
    )

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_ai_comparison_falls_back_for_timeout_and_network_failure(monkeypatch, error_type):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setattr(api, "get_results", saved_results)
    monkeypatch.setitem(
        __import__("sys").modules,
        "groq",
        SimpleNamespace(
            Groq=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: (_ for _ in ()).throw(error_type())
                    )
                )
            )
        ),
    )

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        SimpleNamespace(choices="malformed"),
    ],
)
def test_ai_comparison_falls_back_for_empty_or_malformed_groq_response(monkeypatch, response):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setattr(api, "get_results", saved_results)
    monkeypatch.setitem(
        __import__("sys").modules,
        "groq",
        SimpleNamespace(
            Groq=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_kwargs: response)
                )
            )
        ),
    )

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"


def test_ai_comparison_falls_back_for_unsupported_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unsupported")
    monkeypatch.setenv("GROQ_API_KEY", "backend-test-key")
    monkeypatch.setattr(api, "get_results", saved_results)

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result["source"] == "deterministic-fallback"


def test_ai_endpoints_preserve_response_contract(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN_SECRET", "test-token-secret-that-is-long-enough")
    token = issue_admin_token()
    response_payload = {
        "test_set": "Random holdout",
        "analysis": "Groq analysis",
        "source": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    monkeypatch.setattr(api, "_request_ai_comparison", lambda _payload: response_payload)
    monkeypatch.setattr(api, "get_results", saved_results)

    with TestClient(api.app) as client:
        comparison = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
            headers={"Authorization": f"Bearer {token}"},
        )
        insights = client.post(
            "/api/insights",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert comparison.status_code == 200
    assert comparison.json() == response_payload
    assert insights.status_code == 200
    assert insights.json()["source"] == "groq"
    assert insights.json()["insights"] == "Groq analysis"
