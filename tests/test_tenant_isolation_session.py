"""
Negative tests for tenant isolation of hot session memory (Redis).

Finding C1: the Redis key used to be shrivox:session:{session_id} with a
client-chosen session_id, so two tenants using the same id shared history.
These tests fail if the business_id ever leaves the key.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.memory import session_memory
from app.memory.session_memory import is_valid_session_id

TENANT_A = uuid.uuid4()
TENANT_B = uuid.uuid4()


class FakeRedis:
    """Minimal in-memory stand-in so isolation is tested against real key logic."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture
def fake_redis():
    r = FakeRedis()
    with patch("app.memory.session_memory.get_redis", return_value=r):
        yield r


@pytest.mark.asyncio
async def test_same_session_id_different_tenants_do_not_share_history(fake_redis):
    await session_memory.append_turn(TENANT_A, "shared-id", "customer", "A's card was charged twice")

    assert await session_memory.get_history(TENANT_B, "shared-id") == []
    assert len(await session_memory.get_history(TENANT_A, "shared-id")) == 1


@pytest.mark.asyncio
async def test_tenant_b_writes_do_not_pollute_tenant_a(fake_redis):
    await session_memory.append_turn(TENANT_A, "shared-id", "customer", "from A")
    await session_memory.append_turn(TENANT_B, "shared-id", "customer", "from B")

    a = await session_memory.get_history(TENANT_A, "shared-id")
    b = await session_memory.get_history(TENANT_B, "shared-id")
    assert [t["content"] for t in a] == ["from A"]
    assert [t["content"] for t in b] == ["from B"]


@pytest.mark.asyncio
async def test_clear_session_only_clears_own_tenant(fake_redis):
    await session_memory.append_turn(TENANT_A, "s", "customer", "a")
    await session_memory.append_turn(TENANT_B, "s", "customer", "b")

    await session_memory.clear_session(TENANT_A, "s")

    assert await session_memory.get_history(TENANT_A, "s") == []
    assert len(await session_memory.get_history(TENANT_B, "s")) == 1


def test_key_contains_business_id():
    key = session_memory._key(TENANT_A, "abc")
    assert str(TENANT_A) in key
    assert key != session_memory._key(TENANT_B, "abc")


@pytest.mark.parametrize("sid", ["a", "test-session-1", "A_b-9", "x" * 50])
def test_valid_session_ids(sid):
    assert is_valid_session_id(sid)


@pytest.mark.parametrize("sid", ["", "x" * 51, "a:b", "a b", "../x", "a\nb", "id*", "ünï"])
def test_invalid_session_ids(sid):
    assert not is_valid_session_id(sid)


# ---------- WebSocket level: tenant comes from the verified token only ----------

def _ws_patches(fake_redis, captured):
    async def fake_invoke(state):
        captured.append(state)
        return {
            "response_text": "ok", "intent": "general_query", "escalate": False,
            "safety_flag": None, "retrieved_doc_ids": [], "token_count": 0,
        }

    db_ctx = MagicMock()
    db_ctx.__aenter__ = AsyncMock(return_value=MagicMock(commit=AsyncMock()))
    db_ctx.__aexit__ = AsyncMock(return_value=False)

    return [
        patch("app.api.conversations.AsyncSessionLocal", return_value=db_ctx),
        patch("app.api.conversations._get_or_create_call", new=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4(), status="active"))),
        patch("app.api.conversations._persist_turn", new=AsyncMock()),
        patch("app.api.conversations.shrivox_agent.ainvoke", new=fake_invoke),
        patch("app.memory.session_memory.get_redis", return_value=fake_redis),
    ]


def _token(business_id):
    return create_access_token(uuid.uuid4(), business_id, "admin")


def test_websocket_uses_token_tenant_for_agent_and_memory(fake_redis):
    from app.main import app

    captured: list[dict] = []
    patches = _ws_patches(fake_redis, captured)
    for p in patches:
        p.start()
    try:
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_token(TENANT_A)}") as ws:
            ws.send_json({"text": "hello from A", "business_id": str(TENANT_B)})  # client-supplied id must be ignored
            ws.receive_json()
    finally:
        for p in patches:
            p.stop()

    assert captured[0]["business_id"] == TENANT_A
    assert any(str(TENANT_A) in k for k in fake_redis.store)
    assert not any(str(TENANT_B) in k for k in fake_redis.store)


def test_websocket_tenant_b_does_not_see_tenant_a_history(fake_redis):
    from app.main import app

    captured: list[dict] = []
    patches = _ws_patches(fake_redis, captured)
    for p in patches:
        p.start()
    try:
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_token(TENANT_A)}") as ws:
            ws.send_json({"text": "A's private card issue"})
            ws.receive_json()
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_token(TENANT_B)}") as ws:
            ws.send_json({"text": "B asks something"})
            ws.receive_json()
    finally:
        for p in patches:
            p.stop()

    a_state, b_state = captured
    assert a_state["conversation_history"] == []
    assert b_state["business_id"] == TENANT_B
    assert b_state["conversation_history"] == []  # nothing from tenant A


def test_websocket_rejects_malformed_session_id(fake_redis):
    from app.main import app
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/api/v1/voice/stream/{'x' * 51}?token={_token(TENANT_A)}"):
            pass
    assert exc.value.code == 1008


def test_websocket_rejects_invalid_token(fake_redis):
    from app.main import app
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/v1/voice/stream/s1?token=not-a-jwt"):
            pass
    assert exc.value.code == 1008
