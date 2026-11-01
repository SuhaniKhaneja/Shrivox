"""
Fixtures for tests that run against REAL PostgreSQL (pgvector) and REAL Redis.

Safety rails -- these fixtures TRUNCATE tables and delete Redis keys:
  * the database name must end in `_test`
  * Redis must be logical DB 15
Anything else aborts the run, so a mis-set environment can never wipe a dev
database. Point them with POSTGRES_DSN / REDIS_URL, e.g.

  POSTGRES_DSN=postgresql+asyncpg://shrivox:shrivox@localhost:5432/shrivox_test
  REDIS_URL=redis://localhost:6379/15

If the `_test` database does not exist it is created (needs CREATEDB rights).
Schema comes from the SQLAlchemy models (create_all); once Alembic lands these
tests should also run against migrated schema.
"""

import asyncio
import re
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from integration_helpers import FakeLLM, PerCallRedis, fake_embed_text


def _guard_names() -> str:
    db_name = make_url(settings.postgres_dsn).database or ""
    if not re.fullmatch(r"[a-z0-9_]+_test", db_name):
        pytest.exit(
            f"REFUSING to run integration tests: database '{db_name}' does not end in '_test'. "
            "Set POSTGRES_DSN to a dedicated test database (e.g. .../shrivox_test).", returncode=2)
    if urlparse(settings.redis_url).path != "/15":
        pytest.exit(
            "REFUSING to run integration tests: REDIS_URL must use logical DB 15 "
            "(e.g. redis://localhost:6379/15).", returncode=2)
    return db_name


async def _prepare_schema(db_name: str):
    from app.db.models import Base

    admin = create_async_engine(make_url(settings.postgres_dsn).set(database="postgres"),
                                isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with admin.connect() as conn:
        exists = (await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db_name})).scalar()
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))  # name validated by _guard_names
    await admin.dispose()

    engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    r = Redis.from_url(settings.redis_url)
    try:
        await r.ping()
    finally:
        await r.aclose()


@pytest.fixture(scope="session")
def infrastructure():
    db_name = _guard_names()
    try:
        asyncio.run(_prepare_schema(db_name))
    except Exception as exc:  # opted in, so missing infrastructure is a failure, never a skip
        pytest.fail(
            f"Integration infrastructure unavailable ({type(exc).__name__}: {exc}). "
            "Start PostgreSQL+pgvector and Redis (see README, 'Integration tests').", pytrace=False)


@pytest.fixture(scope="session")
def app_bound_to_test_db(infrastructure):
    """Point the app's own session factory at a NullPool engine on the test DB.
    NullPool matters: asyncpg connections are bound to one event loop and the
    TestClient uses a new loop per request."""
    from app.db.session import AsyncSessionLocal

    original_bind = AsyncSessionLocal.kw["bind"]
    test_engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
    AsyncSessionLocal.configure(bind=test_engine)
    yield
    AsyncSessionLocal.configure(bind=original_bind)
    asyncio.run(test_engine.dispose())


async def _reset_state():
    from app.db.models import Base

    engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()

    r = Redis.from_url(settings.redis_url)
    try:
        async for key in r.scan_iter(match="shrivox:session:*"):
            await r.delete(key)
    finally:
        await r.aclose()


@pytest.fixture(autouse=True)
def clean_state(app_bound_to_test_db, monkeypatch):
    """Every test starts with empty tables, no session keys, fake Gemini
    embeddings, and real Redis reached through a per-call client."""
    asyncio.run(_reset_state())
    monkeypatch.setattr("app.retrieval.retriever.embed_text", fake_embed_text)
    monkeypatch.setattr("app.memory.session_memory.get_redis", lambda: PerCallRedis())


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("app.agents.graph.chat_completion", fake)
    return fake


@pytest.fixture
def client(llm):
    from app.main import app
    return TestClient(app)  # no `with`: startup (CREATE EXTENSION) already done by the harness
