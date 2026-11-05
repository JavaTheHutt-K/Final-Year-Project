"""Small Supabase REST helpers for application settings and experiment runs."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class DatabaseError(RuntimeError):
    """Raised when Supabase is unavailable or returns an invalid response."""


class SupabaseClient:
    """Minimal PostgREST client using the configured service credential."""

    def __init__(self, url: str, key: str) -> None:
        self._url = url.rstrip("/")
        self._key = key

    def request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        upsert: bool = False,
    ) -> list[dict[str, Any]]:
        query = urlencode(params or {})
        if upsert:
            query = f"{query}&on_conflict=setting_key" if query else "on_conflict=setting_key"
        url = f"{self._url}/rest/v1/{table}"
        if query:
            url = f"{url}?{query}"
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if upsert:
            headers["Prefer"] = "resolution=merge-duplicates,return=representation"
        elif method in {"POST", "PATCH"}:
            headers["Prefer"] = "return=representation"
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=15) as response:
                content = response.read()
        except HTTPError as error:
            raise DatabaseError(
                f"Supabase request failed with HTTP {error.code}."
            ) from error
        except (URLError, TimeoutError) as error:
            raise DatabaseError("Could not connect to Supabase.") from error
        if not content:
            return []
        try:
            result = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatabaseError("Supabase returned an invalid response.") from error
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise DatabaseError("Supabase returned an unexpected response shape.")
        return result


def get_supabase_client() -> SupabaseClient:
    """Create a client from backend-only environment credentials."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise DatabaseError("Supabase is not configured; set SUPABASE_URL and SUPABASE_KEY.")
    return SupabaseClient(url, key)


def get_setting(key: str, default: Any = None) -> Any:
    rows = get_supabase_client().request(
        "GET", "app_settings",
        params={"select": "setting_value", "setting_key": f"eq.{key}", "limit": "1"},
    )
    return rows[0].get("setting_value", default) if rows else default


def save_setting(key: str, value: Any) -> None:
    get_supabase_client().request(
        "POST",
        "app_settings",
        params={"select": "setting_key"},
        body={"setting_key": key, "setting_value": value},
        upsert=True,
    )


def create_experiment_run(run: dict[str, Any]) -> dict[str, Any]:
    rows = get_supabase_client().request("POST", "experiment_runs", body=run)
    if not rows:
        raise DatabaseError("Supabase did not return the created experiment run.")
    return rows[0]


def update_experiment_run(run_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
    rows = get_supabase_client().request(
        "PATCH",
        "experiment_runs",
        params={"id": f"eq.{run_id}", "select": "*"},
        body=values,
    )
    return rows[0] if rows else None


def get_experiment_run(run_id: str) -> dict[str, Any] | None:
    rows = get_supabase_client().request(
        "GET",
        "experiment_runs",
        params={"select": "*", "id": f"eq.{run_id}", "limit": "1"},
    )
    return rows[0] if rows else None


def _latest_run(filters: dict[str, str] | None = None) -> dict[str, Any] | None:
    params = {
        "select": "*",
        "order": "created_at.desc",
        "limit": "1",
        **(filters or {}),
    }
    rows = get_supabase_client().request("GET", "experiment_runs", params=params)
    return rows[0] if rows else None


def get_latest_run() -> dict[str, Any] | None:
    return _latest_run()


def get_latest_active_run() -> dict[str, Any] | None:
    return _latest_run({"status": "in.(queued,running)"})


def get_latest_completed_run() -> dict[str, Any] | None:
    return _latest_run({"status": "eq.completed"})


def has_active_experiment() -> bool:
    return get_latest_active_run() is not None
