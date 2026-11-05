from types import SimpleNamespace

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


def test_gemini_comparison_uses_saved_completed_metrics(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "backend-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(api, "get_results", saved_results)
    captured = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="Federated recall is higher.")

    gemini = SimpleNamespace(
        Client=lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "google", SimpleNamespace(genai=gemini))

    result = api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))

    assert result == {
        "test_set": "Random holdout",
        "analysis": "Federated recall is higher.",
    }
    assert captured["model"] == "gemini-2.5-flash"
    assert "Random holdout" in captured["contents"]
    assert "traffic" not in captured["contents"].lower()


def test_ai_comparison_is_rejected_without_admin_authorization():
    with TestClient(api.app) as client:
        response = client.post("/api/ai/compare", json={"test_set": "Random holdout"})
    assert response.status_code == 401


def test_ai_comparison_is_not_available_during_training(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN_SECRET", "test-token-secret-that-is-long-enough")
    monkeypatch.setenv("GEMINI_API_KEY", "backend-test-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "backend-key")
    monkeypatch.setattr(api.database, "get_latest_active_run", lambda: {"id": "run-active"})
    token = issue_admin_token()

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("Gemini must not run while training is active")

    monkeypatch.setattr(api, "_request_ai_comparison", unexpected_request)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 409


def test_ai_comparison_requires_backend_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with __import__("pytest").raises(Exception) as error:
        api._request_ai_comparison(api.AIComparisonRequest(test_set="Random holdout"))
    assert getattr(error.value, "status_code", None) == 503
