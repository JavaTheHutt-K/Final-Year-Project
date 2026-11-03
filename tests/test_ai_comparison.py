import json
from io import BytesIO

from fastapi.testclient import TestClient

from backend import api


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


def test_ai_comparison_uses_saved_metrics_only(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setattr(api, "_load_json", lambda _: saved_results())
    monkeypatch.setitem(api._run_status, "status", "completed")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "Federated recall is higher."}}]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(api, "urlopen", fake_urlopen)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "test_set": "Random holdout",
        "analysis": "Federated recall is higher.",
    }
    assert captured["timeout"] == 30
    request_payload = json.loads(captured["request"].data)
    assert "traffic" not in json.dumps(request_payload).lower()
    assert request_payload["messages"][1]["content"].find("Random holdout") >= 0
    assert captured["request"].get_header("Authorization") == "Bearer test-key"


def test_ai_comparison_is_not_available_during_training(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setitem(api._run_status, "status", "running")

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("AI provider should not be called during training")

    monkeypatch.setattr(api, "urlopen", unexpected_request)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
        )

    assert response.status_code == 409


def test_ai_comparison_requires_backend_api_key(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setitem(api._run_status, "status", "completed")
    with TestClient(api.app) as client:
        response = client.post(
            "/api/ai/compare",
            json={"test_set": "Random holdout"},
        )

    assert response.status_code == 503
    assert "AI_API_KEY" in response.json()["detail"]
