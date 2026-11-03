"""Environment-backed settings reserved for database integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass


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
            database_url=os.environ.get("DATABASE_URL"),
            supabase_url=os.environ.get("SUPABASE_URL"),
            supabase_key=os.environ.get("SUPABASE_KEY"),
        )
