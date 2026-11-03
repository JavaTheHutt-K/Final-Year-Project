import pytest
from fastapi.testclient import TestClient

from backend import api


def test_dashboard_api_is_available_without_authentication():
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/results").status_code == 200
        assert client.get("/api/training-history").status_code == 200
        assert client.get("/api/run/status").status_code == 200


def test_cors_origins_can_be_configured_for_hosted_frontend(monkeypatch):
    monkeypatch.delenv("FRONTEND_URL", raising=False)
    monkeypatch.setenv(
        "IDS_CORS_ORIGINS",
        " https://dashboard.example.com/ , https://preview.example.com ",
    )

    assert api._cors_origins() == [
        "https://dashboard.example.com",
        "https://preview.example.com",
    ]


def test_frontend_url_configures_a_single_exact_origin(monkeypatch):
    monkeypatch.delenv("IDS_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("FRONTEND_URL", " https://dashboard.example.com/ ")

    assert api._cors_origins() == ["https://dashboard.example.com"]


def test_cors_rejects_wildcard_and_non_origin_values(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "*")

    with pytest.raises(ValueError, match="exact http\\(s\\) origins"):
        api._cors_origins()


def test_results_use_configured_temporary_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("IDS_RESULTS_DIR", str(tmp_path))

    assert api._results_path() == tmp_path / "results.json"
    assert api._report_path() == tmp_path / "data_report.json"


def test_authentication_routes_are_removed():
    with TestClient(api.app) as client:
        assert client.get("/api/auth/session").status_code == 404
        assert client.get("/api/auth/google/start").status_code == 404
        assert client.post("/api/auth/logout").status_code == 404
