"""
Negative tests for tenant isolation of hot session memory (Redis).

Finding C1: the Redis key used to be shrivox:session:{session_id} with a
client-chosen session_id, so two tenants using the same id shared history.
These tests fail if the business_id ever leaves the key.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.memory import session_memory
from app.memory.session_memory import is_valid_session_id
from auth_helpers import FakeRedis, forge_token, ws_environment

TENANT_A = uuid.uuid4()
TENANT_B = uuid.uuid4()


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


# ---------- WebSocket level: tenant comes from the verified ticket only ----------
# Tickets are forged with a CHOSEN sid so two tenants can be made to collide on
# the same session id -- something the server never does, but which the Redis
# key scoping must survive.

def _ticket(business_id, sid):
    return forge_token("chat", business_id=business_id, sid=sid)


def test_websocket_uses_ticket_tenant_for_agent_and_memory(fake_redis):
    from app.main import app

    with ws_environment(fake_redis) as env:
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_ticket(TENANT_A, 'shared-id')}") as ws:
            ws.send_json({"text": "hello from A", "business_id": str(TENANT_B)})  # client-supplied id must be ignored
            ws.receive_json()

    assert env.captured[0]["business_id"] == TENANT_A
    assert any(str(TENANT_A) in k for k in fake_redis.store)
    assert not any(str(TENANT_B) in k for k in fake_redis.store)


def test_websocket_tenant_b_does_not_see_tenant_a_history(fake_redis):
    from app.main import app

    with ws_environment(fake_redis) as env:
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_ticket(TENANT_A, 'shared-id')}") as ws:
            ws.send_json({"text": "A's private card issue"})
            ws.receive_json()
        with client.websocket_connect(f"/api/v1/voice/stream/shared-id?token={_ticket(TENANT_B, 'shared-id')}") as ws:
            ws.send_json({"text": "B asks something"})
            ws.receive_json()

    a_state, b_state = env.captured
    assert a_state["conversation_history"] == []
    assert b_state["business_id"] == TENANT_B
    assert b_state["conversation_history"] == []  # nothing from tenant A


def test_websocket_rejects_malformed_session_id_even_with_matching_ticket(fake_redis):
    """Reaches the id-format check: the ticket is valid and its sid matches the
    URL, so only is_valid_session_id can reject this."""
    from structlog.testing import capture_logs
    from app.main import app

    bad_sid = "x" * 51
    with ws_environment(fake_redis) as env, capture_logs() as logs:
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/api/v1/voice/stream/{bad_sid}?token={_ticket(TENANT_A, bad_sid)}"):
                pass
    assert exc.value.code == 1008
    assert [l["reason"] for l in logs if l["event"] == "ws_auth_rejected"] == ["invalid_session_id"]
    env.session_factory.assert_not_called()


def test_websocket_rejects_invalid_token(fake_redis):
    from app.main import app

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/v1/voice/stream/s1?token=not-a-jwt"):
            pass
    assert exc.value.code == 1008
