import sys
from types import SimpleNamespace

import pytest

from backend import cloudrun


def test_launcher_rejects_invalid_run_id():
    with pytest.raises(ValueError, match="valid UUID"):
        cloudrun.launch_training_job("not-a-uuid")


def test_launcher_passes_run_id_and_returns_operation(monkeypatch):
    calls = {}

    class FakeSession:
        def post(self, url, **kwargs):
            calls["url"] = url
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                ok=True,
                status_code=200,
                json=lambda: {"name": "operations/123"},
            )

    fake_google_auth = SimpleNamespace(
        default=lambda **_kwargs: ("credentials", "project")
    )
    fake_transport = SimpleNamespace(AuthorizedSession=lambda _credentials: FakeSession())
    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setattr(google_module, "auth", fake_google_auth, raising=False)
    monkeypatch.setitem(sys.modules, "google.auth", fake_google_auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_transport)
    monkeypatch.setenv("CLOUD_RUN_PROJECT", "project")
    monkeypatch.setenv("CLOUD_RUN_REGION", "africa-south1")
    monkeypatch.setenv("CLOUD_RUN_JOB", "ids-training-job")
    run_id = "00000000-0000-4000-8000-000000000001"

    result = cloudrun.launch_training_job(run_id)

    assert result == {"operation": "operations/123"}
    assert calls["url"].endswith(
        "/projects/project/locations/africa-south1/jobs/ids-training-job:run"
    )
    assert calls["kwargs"]["json"]["overrides"]["containerOverrides"][0]["env"] == [
        {"name": "IDS_RUN_ID", "value": run_id}
    ]
