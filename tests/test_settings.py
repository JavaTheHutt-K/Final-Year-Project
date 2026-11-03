from backend.settings import DatabaseSettings


def test_database_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")

    settings = DatabaseSettings.from_environment()

    assert settings.database_url == "postgresql://example"
    assert settings.supabase_url == "https://example.supabase.co"
    assert settings.supabase_key == "test-key"


def test_database_settings_are_optional(monkeypatch):
    for name in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert DatabaseSettings.from_environment() == DatabaseSettings(None, None, None)
