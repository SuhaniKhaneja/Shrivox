"""
Call lifecycle against a REAL uvicorn server over real sockets.

Why not TestClient: Starlette's WebSocket test session cancels the app task the
instant the client closes, so work the handler does AFTER a disconnect (closing
out the call row) is cut off and cannot be observed. A real ASGI server lets the
handler finish, which is the behaviour production has.

No Gemini call is needed: escalation short-circuits before any LLM or retrieval
call, and connect/disconnect needs no message at all.
"""

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import connect

from app.config import settings
from integration_helpers import db_rows

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def server(infrastructure):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, POSTGRES_DSN=settings.postgres_dsn, REDIS_URL=settings.redis_url,
               JWT_SECRET=settings.jwt_secret, GOOGLE_API_KEY=settings.google_api_key, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--lifespan", "off", "--log-level", "warning"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"uvicorn exited early: {proc.stderr.read().decode()[-500:]}", pytrace=False)
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("uvicorn did not become ready within 30s", pytrace=False)
        yield base, port
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def new_chat(base):
    """Register a tenant and mint a chat ticket, all over real HTTP. Returns (ws_url, business_id)."""
    name = f"live-{uuid.uuid4().hex[:8]}"
    reg = httpx.post(f"{base}/api/v1/auth/register", json={
        "business_name": name, "email": f"{name}@example.com", "password": "correct-horse-battery"})
    assert reg.status_code == 201, reg.text
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    issued = httpx.post(f"{base}/api/v1/auth/chat-ticket", headers=headers).json()
    business_id = db_rows("SELECT id::text AS id FROM businesses WHERE name = :n", n=name)[0]["id"]
    return f"{base.replace('http', 'ws')}{issued['ws_path']}?token={issued['ticket']}", business_id


def call_row(business_id) -> dict | None:
    rows = db_rows("SELECT status, ended_at FROM calls WHERE business_id = :b", b=business_id)
    return rows[0] if rows else None


def wait_for(business_id, predicate, timeout=5.0):
    deadline, row = time.time() + timeout, None
    while time.time() < deadline:
        row = call_row(business_id)
        if row and predicate(row):
            return row
        time.sleep(0.05)
    pytest.fail(f"call row never reached the expected state; last seen: {row}", pytrace=False)


def test_call_is_active_while_connected_then_completed_with_an_end_time(server):
    base, _ = server
    url, biz = new_chat(base)
    with connect(url):
        row = wait_for(biz, lambda r: True)
        assert row["status"] == "active" and row["ended_at"] is None
    row = wait_for(biz, lambda r: r["status"] != "active")
    assert row["status"] == "completed" and row["ended_at"] is not None


def test_an_escalated_call_is_marked_escalated_immediately_and_stays_escalated_after_disconnect(server):
    base, _ = server
    url, biz = new_chat(base)
    with connect(url) as ws:
        ws.send('{"text": "I want to speak to your manager"}')
        import json
        frame = json.loads(ws.recv(timeout=10))
        assert frame["type"] == "response" and frame["escalate"] is True
        assert call_row(biz)["status"] == "escalated"        # saved in the same transaction as the turn
    row = wait_for(biz, lambda r: r["ended_at"] is not None)
    assert row["status"] == "escalated"


def test_reconnecting_to_a_finished_call_reactivates_the_same_call(server):
    base, _ = server
    url, biz = new_chat(base)
    with connect(url):
        wait_for(biz, lambda r: True)
    wait_for(biz, lambda r: r["status"] == "completed")

    with connect(url):                                        # same ticket, still within its lifetime
        row = wait_for(biz, lambda r: r["status"] == "active")
        assert row["ended_at"] is None
    wait_for(biz, lambda r: r["status"] == "completed")
    assert len(db_rows("SELECT 1 FROM calls WHERE business_id = :b", b=biz)) == 1


def test_a_bad_frame_over_a_real_socket_gets_a_typed_error_and_the_connection_survives(server):
    import json
    base, _ = server
    url, biz = new_chat(base)
    with connect(url) as ws:
        ws.send("this is not json")
        frame = json.loads(ws.recv(timeout=10))
        assert frame == {"type": "error", "code": "invalid_frame", "retryable": False,
                         "message": frame["message"]}
        ws.send('{"text": "   "}')
        assert json.loads(ws.recv(timeout=10))["code"] == "invalid_message"
        ws.send('{"text": "I want to speak to your manager"}')   # same connection still serves real turns
        assert json.loads(ws.recv(timeout=10))["type"] == "response"
