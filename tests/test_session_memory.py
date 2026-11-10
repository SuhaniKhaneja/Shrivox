import json
import uuid
from unittest.mock import patch

import pytest

from app.config import settings
from app.memory import session_memory
from auth_helpers import FakeRedis

TENANT_A = uuid.uuid4()


@pytest.fixture(autouse=True)
def reset_redis_singleton():
    session_memory._redis = None
    yield
    session_memory._redis = None


@pytest.fixture
def redis():
    r = FakeRedis()
    with patch("app.memory.session_memory.get_redis", return_value=r):
        yield r


def _seed(redis, session_id, messages):
    redis.store[session_memory._key(TENANT_A, session_id)] = [json.dumps(m) for m in messages]


@pytest.mark.asyncio
async def test_append_turn_grows_history(redis):
    _seed(redis, "session-1", [{"role": "customer", "content": "hi"}])

    history = await session_memory.append_turn(TENANT_A, "session-1", "agent", "hello there")

    assert len(history) == 2
    assert history[-1] == {"role": "agent", "content": "hello there"}


@pytest.mark.asyncio
async def test_history_capped_at_max_turns(redis):
    full = [{"role": "customer" if i % 2 == 0 else "agent", "content": f"msg {i}"}
            for i in range(settings.max_conversation_turns)]
    _seed(redis, "session-1", full)

    history = await session_memory.append_turns(
        TENANT_A, "session-1", [("customer", "one more question"), ("agent", "one more answer")])

    assert len(history) == settings.max_conversation_turns
    assert history[-1]["content"] == "one more answer"
    assert "msg 0" not in [m["content"] for m in history]   # the oldest pair was dropped


@pytest.mark.asyncio
async def test_empty_history_returns_empty_list(redis):
    assert await session_memory.get_history(TENANT_A, "new-session") == []


@pytest.mark.asyncio
async def test_a_turn_is_written_in_one_atomic_pipeline_with_ttl(redis):
    await session_memory.append_turns(TENANT_A, "s1", [("customer", "q"), ("agent", "a")])

    assert redis.executed_pipelines == 1                                   # one MULTI/EXEC, not several writes
    key = session_memory._key(TENANT_A, "s1")
    assert [json.loads(x)["role"] for x in redis.store[key]] == ["customer", "agent"]
    assert redis.ttls[key] == settings.session_ttl_seconds


@pytest.mark.asyncio
async def test_odd_cap_cannot_leave_an_orphaned_reply_at_the_head(redis, monkeypatch):
    """A cap that cuts between a customer message and its reply must not hand the
    model a history that starts with its own answer (Gemini rejects that)."""
    monkeypatch.setattr(settings, "max_conversation_turns", 3)
    for i in range(2):
        history = await session_memory.append_turns(TENANT_A, "s", [("customer", f"q{i}"), ("agent", f"a{i}")])

    assert history[0]["role"] == "customer"
    assert [m["content"] for m in history] == ["q1", "a1"]   # stored window was [a0, q1, a1]; the orphan is dropped


@pytest.mark.asyncio
async def test_corrupt_entry_raises_value_error_so_callers_degrade(redis):
    redis.store[session_memory._key(TENANT_A, "bad")] = ["{not json"]
    with pytest.raises(ValueError):
        await session_memory.get_history(TENANT_A, "bad")


@pytest.mark.asyncio
async def test_clear_session_also_removes_a_legacy_string_key(redis):
    redis.store[session_memory._legacy_key(TENANT_A, "s")] = '[{"role": "customer", "content": "old"}]'
    await session_memory.append_turn(TENANT_A, "s", "customer", "new")

    await session_memory.clear_session(TENANT_A, "s")

    assert redis.store == {}


def test_v2_prefix_keeps_legacy_string_keys_out_of_the_way():
    """A legacy STRING under the old key would raise WRONGTYPE if read as a list."""
    assert session_memory._key(TENANT_A, "s") != session_memory._legacy_key(TENANT_A, "s")
    assert session_memory._key(TENANT_A, "s").startswith("shrivox:session:v2:")


@pytest.mark.parametrize("bad", [0, 1, 501])
def test_message_cap_is_bounded_so_ltrim_can_never_be_disabled(bad):
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None, google_api_key="k", jwt_secret="s", max_conversation_turns=bad)
