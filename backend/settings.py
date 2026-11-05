"""Environment-backed settings reserved for database integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_env(name: str) -> str | None:
    """Return a trimmed environment value or None when the variable is blank."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class DatabaseSettings:
    """Optional database credentials supplied by the deployment environment."""

    database_url: str | None
    supabase_url: str | None
    supabase_key: str | None

    @classmethod
    def from_environment(cls) -> DatabaseSettings:
        """Read optional connection settings without logging or persisting them."""
        return cls(
            database_url=_optional_env("DATABASE_URL"),
            supabase_url=_optional_env("SUPABASE_URL"),
            supabase_key=_optional_env("SUPABASE_KEY"),
        )
