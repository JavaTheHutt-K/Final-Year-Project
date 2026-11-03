import sys
from concurrent.futures import Future, ThreadPoolExecutor
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
        assert client.get("/api/run/status").status_code == 200


def test_gpu_check_is_registered_and_uses_cuda_availability(monkeypatch):
    assert any(
        function.fn is check_gpu_availability
        for function in demo.fns.values()
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert check_gpu_availability() == {"cuda_available": False}


def test_gradio_start_button_uses_existing_background_task(monkeypatch):
    submissions = []

    def fake_submit(function, *args, **kwargs):
        future = Future()
        future.set_result(None)
        submissions.append((function, args, kwargs))
        return future

    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(backend_api, "_executor", executor)
    monkeypatch.setattr(executor, "submit", fake_submit)
    monkeypatch.setattr(backend_api, "_run_future", None)
    monkeypatch.setattr(backend_api, "_run_status", {
        "status": "idle",
        "stage": "idle",
        "message": "Ready to run the experiment.",
    })
    monkeypatch.setattr(backend_api, "_live_training_history", [])

    registered_handler = next(
        function.fn
        for function in demo.fns.values()
        if function.fn is start_experiment
    )
    response, status = registered_handler()

    assert response == {
        "status": "running",
        "message": "Experiment started in the background.",
    }
    assert status["status"] == "running"
    assert submissions == [(backend_api._execute_experiment, (), {})]

    with TestClient(app) as client:
        api_response = client.post("/api/run")

    assert api_response.status_code == 202
    assert api_response.json() == response
    assert submissions == [
        (backend_api._execute_experiment, (), {}),
        (backend_api._execute_experiment, (), {}),
    ]
