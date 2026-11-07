"""
Concurrency tests against REAL Redis and REAL PostgreSQL.

These exist because the failures they catch are invisible to mocks: a fake that
is "atomic" by construction proves nothing about Redis, and a mocked session
cannot lose a race. Each test here was written to FAIL against the code it
replaced (see the audit notes in docs/AUDIT.md) and pass against the fix.
"""

import asyncio
import uuid

import pytest
from redis.asyncio import Redis

from app.config import settings
from app.memory import session_memory

pytestmark = pytest.mark.integration


def _run_with_shared_redis(monkeypatch, coro_factory):
    """Run `coro_factory()` on one event loop with ONE real Redis client, which is
    what a production worker has. (The suite's default per-call client would hide
    interleaving by serialising nothing.)"""
    async def go():
        client = Redis.from_url(settings.redis_url, decode_responses=True)
        monkeypatch.setattr("app.memory.session_memory.get_redis", lambda: client)
        try:
            return await coro_factory()
        finally:
            await client.aclose()
    return asyncio.run(go())


# ------------------------------------------------------------ session history

def test_concurrent_appends_to_one_session_lose_nothing(monkeypatch):
    """Two tabs, or a reconnect overlapping an in-flight turn, append to the same
    session at once. The old read-modify-write of one JSON blob kept ~1 of 30."""
    monkeypatch.setattr(settings, "max_conversation_turns", 1000)
    biz, turns = uuid.uuid4(), 40

    async def scenario():
        await asyncio.gather(*[
            session_memory.append_turns(biz, "shared-session", [("customer", f"q{i}"), ("agent", f"a{i}")])
            for i in range(turns)
        ])
        return await session_memory.get_history(biz, "shared-session")

    history = _run_with_shared_redis(monkeypatch, scenario)

    assert len(history) == turns * 2, f"lost {turns - len(history) // 2} of {turns} turns"
    # A customer message and its reply are written together, so they stay adjacent
    # even when many turns interleave: pairs are never split or crossed.
    for i in range(0, len(history), 2):
        assert history[i]["role"] == "customer" and history[i + 1]["role"] == "agent"
        assert history[i + 1]["content"] == "a" + history[i]["content"][1:]


def test_window_is_enforced_atomically_under_concurrency(monkeypatch):
    monkeypatch.setattr(settings, "max_conversation_turns", 10)
    biz = uuid.uuid4()

    async def scenario():
        await asyncio.gather(*[
            session_memory.append_turns(biz, "s", [("customer", f"q{i}"), ("agent", f"a{i}")]) for i in range(40)
        ])
        return await session_memory.get_history(biz, "s")

    history = _run_with_shared_redis(monkeypatch, scenario)
    assert len(history) == 10
    for i in range(0, 10, 2):  # whatever survives is still whole pairs, customer first
        assert history[i]["role"] == "customer" and history[i + 1]["content"] == "a" + history[i]["content"][1:]


def test_session_key_expires(monkeypatch):
    monkeypatch.setattr(settings, "session_ttl_seconds", 120)
    biz = uuid.uuid4()

    async def scenario():
        await session_memory.append_turns(biz, "ttl", [("customer", "hi"), ("agent", "hello")])
        return await session_memory.get_redis().ttl(session_memory._key(biz, "ttl"))

    ttl = _run_with_shared_redis(monkeypatch, scenario)
    assert 0 < ttl <= 120
