"""Administrator password verification and short-lived signed bearer tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _token_secret() -> bytes:
    secret = os.environ.get("ADMIN_TOKEN_SECRET", "").encode()
    if len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Administrator authentication is not configured.",
        )
    return secret


def verify_admin_password(password: str) -> bool:
    encoded = os.environ.get("ADMIN_PASSWORD_HASH", "")
    try:
        algorithm, rounds, salt, digest = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = _b64decode(digest)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _b64decode(salt), int(rounds)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def issue_admin_token() -> str:
    now = int(time.time())
    header = _b64encode(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64encode(
        json.dumps(
            {"sub": "admin", "iat": now, "exp": now + 3600},
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    signature = hmac.new(_token_secret(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64encode(signature)}"


def validate_admin_token(token: str) -> bool:
    try:
        header, payload, signature = token.split(".")
        signing_input = f"{header}.{payload}".encode()
        expected = hmac.new(_token_secret(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature)):
            return False
        claims: dict[str, Any] = json.loads(_b64decode(payload))
        return claims.get("sub") == "admin" and int(claims["exp"]) > int(time.time())
    except (HTTPException, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if credentials is None or not validate_admin_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Administrator authentication required.")


def verify_gradio_admin(username: str | None) -> None:
    configured_username = os.environ.get("GRADIO_ADMIN_USERNAME", "")
    if not configured_username or not username or not hmac.compare_digest(
        username, configured_username
    ):
        raise HTTPException(status_code=401, detail="Administrator authentication required.")


def verify_gradio_credentials(username: str, password: str) -> bool:
    configured_username = os.environ.get("GRADIO_ADMIN_USERNAME", "")
    return bool(
        configured_username
        and hmac.compare_digest(username, configured_username)
        and verify_admin_password(password)
    )
