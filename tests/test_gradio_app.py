import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import app, check_gpu_availability, demo, start_experiment
from backend import api as backend_api


def test_gradio_interface_and_fastapi_endpoints_share_the_app():
    assert demo.ssr_mode is False
    assert getattr(demo, "node_process", None) is None

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/run/status").status_code == 503


def test_gpu_check_is_registered_and_uses_cuda_availability(monkeypatch):
    assert any(function.fn is check_gpu_availability for function in demo.fns.values())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert check_gpu_availability() == {"cuda_available": False}


def test_gradio_and_api_start_actions_share_orchestration(monkeypatch):
    calls = []

    def fake_start():
        calls.append("start")
        return {"run_id": "run-123", "status": "queued"}

    monkeypatch.setattr(backend_api, "start_run_for_authorized_admin", fake_start)
    monkeypatch.setattr(
        backend_api,
        "get_run_status",
        lambda: {"run_id": "run-123", "status": "queued"},
    )
    monkeypatch.setenv("GRADIO_ADMIN_USERNAME", "admin")
    registered_handler = next(function.fn for function in demo.fns.values() if function.fn is start_experiment)
    response, status = registered_handler(SimpleNamespace(username="admin"))

    assert response == {"run_id": "run-123", "status": "queued"}
    assert status["status"] == "queued"
    assert calls == ["start"]


def test_gradio_start_action_rejects_anonymous_user(monkeypatch):
    registered_handler = next(function.fn for function in demo.fns.values() if function.fn is start_experiment)
    try:
        registered_handler(SimpleNamespace(username=None))
    except Exception as error:
        assert getattr(error, "status_code", None) == 401
    else:
        raise AssertionError("Anonymous Gradio launch should be rejected.")
