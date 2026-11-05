"""Launch the configured Google Cloud Run Job for an experiment run."""

from __future__ import annotations

import os
from urllib.parse import quote


def launch_training_job(run_id: str) -> dict[str, str]:
    """Start a Cloud Run Job execution and return its operation identifier."""
    project = os.environ.get("CLOUD_RUN_PROJECT", "").strip()
    region = os.environ.get("CLOUD_RUN_REGION", "").strip()
    job = os.environ.get("CLOUD_RUN_JOB", "").strip()
    if not project or not region or not job:
        raise RuntimeError(
            "Cloud Run Job is not configured; set CLOUD_RUN_PROJECT, "
            "CLOUD_RUN_REGION, and CLOUD_RUN_JOB."
        )

    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    resource = (
        f"projects/{quote(project, safe='')}/locations/{quote(region, safe='')}"
        f"/jobs/{quote(job, safe='')}"
    )
    response = session.post(
        f"https://run.googleapis.com/v2/{resource}:run",
        json={
            "overrides": {
                "containerOverrides": [
                    {"env": [{"name": "IDS_RUN_ID", "value": run_id}]}
                ]
            }
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Cloud Run Job launch failed with HTTP {response.status_code}."
        )
    payload = response.json()
    operation = payload.get("name")
    if not isinstance(operation, str) or not operation:
        raise RuntimeError("Cloud Run returned no execution operation name.")
    return {"operation": operation}
