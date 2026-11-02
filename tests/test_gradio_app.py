from concurrent.futures import Future

from fastapi.testclient import TestClient

from app import app, demo, start_experiment
from backend import api as backend_api


def test_gradio_interface_and_fastapi_endpoints_share_the_app():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/run/status").status_code == 200


def test_gradio_start_button_uses_existing_background_task(monkeypatch):
    submissions = []

    def fake_submit(function, *args, **kwargs):
        future = Future()
        future.set_result(None)
        submissions.append((function, args, kwargs))
        return future

    monkeypatch.setattr(backend_api._executor, "submit", fake_submit)
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
