"""scripts/test_chat.py must obtain its credential through the real, authenticated
flow (login -> /auth/chat-ticket). Runs the script's own HTTP helper against the
real FastAPI app in-process; only the DB lookup of the user is faked."""

import importlib.util
import pathlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.security import SCOPE_CHAT, SCOPE_DASHBOARD, decode_token, hash_password
from app.db.session import get_db
from app.main import app

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "test_chat.py"
BUSINESS_ID, USER_ID = uuid.uuid4(), uuid.uuid4()


def load_script():
    spec = importlib.util.spec_from_file_location("test_chat_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def user_db():
    user = SimpleNamespace(id=USER_ID, business_id=BUSINESS_ID, role="admin",
                           email="owner@example.com", hashed_password=hash_password("correct-horse-battery"))

    async def fake_get_db():
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=user)))
        yield session

    app.dependency_overrides[get_db] = fake_get_db
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_script_gets_a_chat_ticket_via_login_then_ticket_endpoint(user_db):
    script = load_script()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        issued = await script.fetch_chat_ticket(client, "owner@example.com", "correct-horse-battery")

    claims = decode_token(issued["ticket"], SCOPE_CHAT)
    assert claims.business_id == BUSINESS_ID and claims.sid == issued["session_id"]
    with pytest.raises(Exception):  # the script must hand the WebSocket a ticket, not a dashboard token
        decode_token(issued["ticket"], SCOPE_DASHBOARD)
    url = script.websocket_url("http://localhost:8000", issued)
    assert url.startswith(f"ws://localhost:8000/api/v1/voice/stream/{issued['session_id']}?token=")


@pytest.mark.asyncio
async def test_script_surfaces_login_failure_without_ticket(user_db):
    script = load_script()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await script.fetch_chat_ticket(client, "owner@example.com", "wrong-password")
    assert exc.value.response.status_code == 401
