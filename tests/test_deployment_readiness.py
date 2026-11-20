from types import SimpleNamespace

import pytest

from app.db.session import database_url_and_options
from scripts.seed_demo_business import load_demo_credentials


def _db_config(**overrides):
    values = {
        "postgres_dsn": "postgresql://demo:secret@ep-example.neon.tech/demo?sslmode=require&channel_binding=require",
        "database_require_tls": True,
        "database_connect_timeout_seconds": 10,
        "database_pool_size": 1,
        "database_max_overflow": 0,
        "database_pool_timeout_seconds": 15,
        "database_pool_recycle_seconds": 240,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_neon_url_is_normalized_for_asyncpg_and_tls():
    url, options = database_url_and_options(_db_config())

    assert url.drivername == "postgresql+asyncpg"
    assert "sslmode" not in url.query
    assert "channel_binding" not in url.query
    assert options["connect_args"] == {"timeout": 10, "ssl": "require"}
    assert options["pool_size"] == 1
    assert options["max_overflow"] == 0
    assert options["pool_pre_ping"] is True
    assert options["pool_recycle"] == 240


def test_libpq_sslmode_is_honoured_when_tls_flag_is_not_set():
    config = _db_config(database_require_tls=False)
    url, options = database_url_and_options(config)

    assert url.drivername == "postgresql+asyncpg"
    assert options["connect_args"]["ssl"] == "require"


def test_tls_requirement_rejects_explicit_ssl_disable():
    config = _db_config(postgres_dsn="postgresql://demo:secret@db.example/demo?sslmode=disable")
    with pytest.raises(ValueError, match="conflicts"):
        database_url_and_options(config)


def test_non_postgres_dsn_is_rejected():
    config = _db_config(postgres_dsn="sqlite+aiosqlite:///./demo.db", database_require_tls=False)
    with pytest.raises(ValueError, match="requires the asyncpg driver"):
        database_url_and_options(config)


def test_demo_seed_requires_credentials():
    with pytest.raises(RuntimeError, match="DEMO_EMAIL and DEMO_PASSWORD"):
        load_demo_credentials({})


def test_demo_seed_normalizes_email_and_validates_password():
    email, password, business = load_demo_credentials({
        "DEMO_EMAIL": " Demo.User@example.com ",
        "DEMO_PASSWORD": "a-unique-demo-passphrase",
    })
    assert email == "demo.user@example.com"
    assert password == "a-unique-demo-passphrase"
    assert business == "Demo E-Commerce Co"


def test_demo_seed_rejects_short_password():
    with pytest.raises(ValueError, match="at least 12"):
        load_demo_credentials({"DEMO_EMAIL": "demo@example.com", "DEMO_PASSWORD": "short"})


def test_demo_seed_rejects_password_over_bcrypt_limit():
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        load_demo_credentials({"DEMO_EMAIL": "demo@example.com", "DEMO_PASSWORD": "x" * 73})


def test_metrics_route_is_disabled_when_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "metrics_enabled", False)
    response = TestClient(app).get("/metrics")
    assert response.status_code == 404
