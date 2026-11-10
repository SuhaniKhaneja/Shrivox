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
from sqlalchemy import select

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


# ------------------------------------------------------- call get-or-create

async def _new_business(name="race-tenant"):
    from app.db.models import Business
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        business = Business(id=uuid.uuid4(), name=name)
        db.add(business)
        await db.commit()
        return business.id


async def _race_connects(business_id, session_id, racers=20):
    """Open `racers` sessions, each already holding a live connection, then run
    _get_or_create_call on all of them at once -- what N simultaneous WebSocket
    connects with one (replayable) chat ticket look like. Pre-warming matters: with
    a cold pool, connection setup serialises the racers and hides the race."""
    from app.api.conversations import _get_or_create_call
    from app.db.session import AsyncSessionLocal

    sessions = [AsyncSessionLocal() for _ in range(racers)]
    try:
        for s in sessions:
            await s.connection()
        calls = await asyncio.gather(*[_get_or_create_call(s, business_id, session_id) for s in sessions])
        return [c.id for c in calls]
    finally:
        for s in sessions:
            await s.close()


def test_concurrent_connects_with_one_session_id_create_exactly_one_call():
    from app.db.models import Call
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import func, select

    async def scenario():
        business_id = await _new_business()
        sid = "race-" + uuid.uuid4().hex[:12]
        ids = await _race_connects(business_id, sid)
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(func.count()).select_from(Call).where(Call.business_id == business_id, Call.caller_ref == sid)
            )).scalar_one()
        return ids, rows

    ids, rows = asyncio.run(scenario())
    assert rows == 1, f"{rows} Call rows were created for one session"
    assert len(set(ids)) == 1, "racers were handed different calls"


def test_a_session_that_already_has_duplicate_calls_still_reconnects():
    """Databases that hit the race before the fix hold duplicate rows. Those
    sessions used to raise MultipleResultsFound on EVERY later connect; they must
    heal by resolving to the earliest call."""
    from datetime import datetime, timedelta
    from app.api.conversations import _get_or_create_call
    from app.db.models import Call
    from app.db.session import AsyncSessionLocal

    async def scenario():
        business_id = await _new_business()
        sid = "legacy-" + uuid.uuid4().hex[:12]
        base = datetime.utcnow()
        async with AsyncSessionLocal() as db:
            for offset in (0, 1, 2):
                db.add(Call(id=uuid.uuid4(), business_id=business_id, channel="browser", caller_ref=sid,
                            status="active", started_at=base + timedelta(seconds=offset)))
            await db.commit()
            first = (await db.execute(
                select(Call.id).where(Call.caller_ref == sid).order_by(Call.started_at))).scalars().first()
        async with AsyncSessionLocal() as db:
            call = await _get_or_create_call(db, business_id, sid)
        return first, call.id

    earliest, resolved = asyncio.run(scenario())
    assert resolved == earliest


def test_get_or_create_does_not_hold_its_lock_after_returning():
    """The advisory lock is transaction-scoped. If the function returned without
    ending its transaction, the lock would be held for the whole WebSocket
    connection and every other connect for that session would hang."""
    from app.api.conversations import _get_or_create_call
    from app.db.session import AsyncSessionLocal

    async def scenario():
        business_id = await _new_business()
        sid = "lock-" + uuid.uuid4().hex[:12]
        async with AsyncSessionLocal() as creator, AsyncSessionLocal() as finder, AsyncSessionLocal() as third:
            await _get_or_create_call(creator, business_id, sid)   # create path; session stays open
            await _get_or_create_call(finder, business_id, sid)    # existing-call path; session stays open
            # Both sessions above are still open, as they are for a live WebSocket. A third
            # connection must not wait on a lock either of them forgot to release.
            return await asyncio.wait_for(_get_or_create_call(third, business_id, sid), timeout=5)

    asyncio.run(scenario())  # a held lock would raise asyncio.TimeoutError here


def test_session_key_expires(monkeypatch):
    monkeypatch.setattr(settings, "session_ttl_seconds", 120)
    biz = uuid.uuid4()

    async def scenario():
        await session_memory.append_turns(biz, "ttl", [("customer", "hi"), ("agent", "hello")])
        return await session_memory.get_redis().ttl(session_memory._key(biz, "ttl"))

    ttl = _run_with_shared_redis(monkeypatch, scenario)
    assert 0 < ttl <= 120
