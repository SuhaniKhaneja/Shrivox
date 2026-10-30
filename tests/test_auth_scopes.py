"""
Scoped-credential tests (audit H2).

Boundaries under test:
  * a dashboard token authorizes REST and is refused by the WebSocket
  * a chat ticket authorizes exactly one WebSocket session and is refused by REST
  * tickets are short-lived, session-bound and tenant-bound
  * tenant identity and session id can never be chosen by the client
  * malformed / expired / incomplete / forged tokens are rejected everywhere
  * credentials never reach the logs

Happy paths use the production issuers (create_access_token and
POST /auth/chat-ticket). forge_token is used only to build tokens the server
would never issue.
"""

import base64
import json
import logging
import re
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect
from structlog.testing import capture_logs

from app.config import Settings, settings
from app.core.log_redaction import RedactTokenFilter, install_token_redaction, redact_token
from app.core.security import (
    SCOPE_CHAT,
    SCOPE_DASHBOARD,
    InsufficientScopeError,
    TokenError,
    create_access_token,
    create_chat_ticket,
    decode_token,
)
from app.db.session import get_db
from app.main import app
from auth_helpers import FakeRedis, assert_untouched, forge_token, ws_environment

TENANT_A = uuid.uuid4()
TENANT_B = uuid.uuid4()
USER_A = uuid.uuid4()


# ----------------------------- fixtures ---------------------------------

class _FakeResult:
    def scalars(self):
        return self

    def all(self):
        return []

    def scalar_one_or_none(self):
        return None


class _FakeSession:
    def __init__(self):
        self.execute = AsyncMock(return_value=_FakeResult())
        self.add = MagicMock()
        self.commit = AsyncMock()
        self.refresh = AsyncMock()
        self.delete = AsyncMock()


@pytest.fixture
def client():
    async def fake_get_db():
        yield _FakeSession()

    app.dependency_overrides[get_db] = fake_get_db
    with patch("app.api.documents.upsert_chunk", new=AsyncMock()):  # never call the embedding API
        yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def fake_redis():
    return FakeRedis()


def dashboard_token(business_id=TENANT_A, user_id=USER_A, role="admin"):
    return create_access_token(user_id, business_id, role)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


ENDPOINTS = [
    ("GET", "/api/v1/documents", None),
    ("POST", "/api/v1/documents/upload", {"filename": "a.txt", "text": "hello"}),
    ("DELETE", f"/api/v1/documents/{uuid.uuid4()}", None),
    ("GET", "/api/v1/conversations", None),
    ("GET", f"/api/v1/conversations/{uuid.uuid4()}", None),
    ("POST", "/api/v1/auth/chat-ticket", None),
]
ENDPOINT_IDS = ["list_documents", "upload_document", "delete_document", "list_conversations", "get_conversation", "issue_chat_ticket"]


def call(client, method, path, body, headers=None):
    return client.request(method, path, json=body, headers=headers or {})


# ------------------- REST: dashboard in, chat ticket out ----------------

@pytest.mark.parametrize("method,path,body", ENDPOINTS, ids=ENDPOINT_IDS)
def test_dashboard_token_is_authorized_on_protected_endpoints(client, method, path, body):
    r = call(client, method, path, body, bearer(dashboard_token()))
    # 404 is fine for delete/get-by-id: auth passed, the (fake) row just doesn't exist
    assert r.status_code in (200, 201, 404), r.text


@pytest.mark.parametrize("method,path,body", ENDPOINTS, ids=ENDPOINT_IDS)
def test_chat_ticket_is_rejected_on_protected_endpoints(client, method, path, body):
    ticket = create_chat_ticket(USER_A, TENANT_A).ticket
    r = call(client, method, path, body, bearer(ticket))
    assert r.status_code == 403


@pytest.mark.parametrize("method,path,body", ENDPOINTS, ids=ENDPOINT_IDS)
def test_missing_credentials_are_rejected_with_401(client, method, path, body):
    r = call(client, method, path, body)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def _tamper_payload(token: str, **changes) -> str:
    header, payload, sig = token.split(".")
    data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    data.update(changes)
    new_payload = base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
    return f"{header}.{new_payload}.{sig}"


BAD_DASHBOARD_TOKENS = {
    "garbage": lambda: "not-a-jwt",
    "empty": lambda: "x",
    "expired": lambda: forge_token("dashboard", ttl=-10),
    "wrong_secret": lambda: forge_token("dashboard", secret="some-other-secret-value-0123456789"),
    "alg_none": lambda: forge_token("dashboard", secret="", algorithm="none"),
    "legacy_no_scope": lambda: forge_token("dashboard", drop=("scope",)),
    "no_exp": lambda: forge_token("dashboard", drop=("exp",)),
    "no_iat": lambda: forge_token("dashboard", drop=("iat",)),
    "no_sub": lambda: forge_token("dashboard", drop=("sub",)),
    "no_business_id": lambda: forge_token("dashboard", drop=("business_id",)),
    "non_uuid_sub": lambda: forge_token("dashboard", overrides={"sub": "admin"}),
    "non_uuid_business": lambda: forge_token("dashboard", overrides={"business_id": "1 OR 1=1"}),
    "bad_role": lambda: forge_token("dashboard", role="superuser"),
    "no_role": lambda: forge_token("dashboard", drop=("role",)),
    "unknown_scope": lambda: forge_token("dashboard", overrides={"scope": "admin"}),
    "tampered_business_id": lambda: _tamper_payload(dashboard_token(TENANT_A), business_id=str(TENANT_B)),
}


@pytest.mark.parametrize("name", BAD_DASHBOARD_TOKENS)
def test_invalid_dashboard_tokens_are_rejected_on_rest(client, name):
    r = client.get("/api/v1/documents", headers=bearer(BAD_DASHBOARD_TOKENS[name]()))
    assert r.status_code in (401, 403), f"{name} -> {r.status_code}"
    # A structurally bad token must be 401 (not authenticated); only a genuine,
    # correctly signed token of another scope earns 403.
    if name != "unknown_scope":
        assert r.status_code == 401


# --------------------- chat-ticket minting endpoint ---------------------

def test_minted_ticket_is_bound_to_tenant_and_session(client):
    r = client.post("/api/v1/auth/chat-ticket", headers=bearer(dashboard_token(TENANT_A, USER_A)))
    assert r.status_code == 201
    body = r.json()

    claims = decode_token(body["ticket"], expected_scope=SCOPE_CHAT)
    assert claims.business_id == TENANT_A
    assert claims.user_id == USER_A          # issuing dashboard user (audit trail)
    assert claims.sid == body["session_id"]
    assert claims.role is None               # a ticket carries no dashboard role
    assert body["expires_in"] == settings.chat_ticket_ttl_seconds
    assert body["ws_path"] == f"/api/v1/voice/stream/{body['session_id']}"


def test_ticket_request_cannot_choose_tenant_session_or_scope(client):
    hostile = {"business_id": str(TENANT_B), "session_id": "attacker-chosen", "scope": "dashboard", "sid": "x"}
    r = client.post("/api/v1/auth/chat-ticket", json=hostile, headers=bearer(dashboard_token(TENANT_A)))
    assert r.status_code == 201
    claims = decode_token(r.json()["ticket"], expected_scope=SCOPE_CHAT)
    assert claims.business_id == TENANT_A
    assert claims.sid != "attacker-chosen"
    with pytest.raises(InsufficientScopeError):
        decode_token(r.json()["ticket"], expected_scope=SCOPE_DASHBOARD)


def test_session_ids_are_unique_and_unguessable():
    ids = {create_chat_ticket(USER_A, TENANT_A).session_id for _ in range(500)}
    assert len(ids) == 500
    for sid in ids:
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", sid)  # 32 random bytes, urlsafe-b64


def test_ticket_lifetime_follows_configuration(monkeypatch):
    monkeypatch.setattr(settings, "chat_ticket_ttl_seconds", 7)
    issued = create_chat_ticket(USER_A, TENANT_A)
    raw = jwt.decode(issued.ticket, options={"verify_signature": False})
    assert raw["exp"] - raw["iat"] == 7
    assert issued.expires_in == 7


def test_default_ticket_lifetime_is_short():
    raw = jwt.decode(create_chat_ticket(USER_A, TENANT_A).ticket, options={"verify_signature": False})
    assert raw["exp"] - raw["iat"] <= 60
    # and far shorter than a dashboard session
    dash = jwt.decode(dashboard_token(), options={"verify_signature": False})
    assert dash["exp"] - dash["iat"] > 60 * 60


@pytest.mark.parametrize("ttl,ok", [(0, False), (4, False), (5, True), (900, True), (901, False), (86400, False)])
def test_ticket_lifetime_setting_is_bounded(ttl, ok):
    build = lambda: Settings(_env_file=None, google_api_key="k", jwt_secret="s", chat_ticket_ttl_seconds=ttl)
    if ok:
        assert build().chat_ticket_ttl_seconds == ttl
    else:
        with pytest.raises(ValidationError):
            build()


# ------------------------- decode_token unit ----------------------------

def test_decode_token_enforces_scope_both_ways():
    dash, chat = dashboard_token(), create_chat_ticket(USER_A, TENANT_A).ticket
    assert decode_token(dash, SCOPE_DASHBOARD).scope == SCOPE_DASHBOARD
    assert decode_token(chat, SCOPE_CHAT).scope == SCOPE_CHAT
    with pytest.raises(InsufficientScopeError):
        decode_token(dash, SCOPE_CHAT)
    with pytest.raises(InsufficientScopeError):
        decode_token(chat, SCOPE_DASHBOARD)


BAD_CHAT_TICKETS = {
    "garbage": lambda: "not-a-jwt",
    "expired": lambda: forge_token("chat", ttl=-10),
    "wrong_secret": lambda: forge_token("chat", secret="some-other-secret-value-0123456789"),
    "alg_none": lambda: forge_token("chat", secret="", algorithm="none"),
    "no_scope": lambda: forge_token("chat", drop=("scope",)),
    "no_sid": lambda: forge_token("chat", drop=("sid",)),
    "empty_sid": lambda: forge_token("chat", sid=""),
    "non_string_sid": lambda: forge_token("chat", overrides={"sid": 12345}),
    "no_exp": lambda: forge_token("chat", drop=("exp",)),
    "no_iat": lambda: forge_token("chat", drop=("iat",)),
    "no_business_id": lambda: forge_token("chat", drop=("business_id",)),
    "non_uuid_business": lambda: forge_token("chat", overrides={"business_id": "tenant-a"}),
    "no_sub": lambda: forge_token("chat", drop=("sub",)),
    "scope_dashboard_with_sid": lambda: forge_token("chat", overrides={"scope": "dashboard"}),
}


@pytest.mark.parametrize("name", BAD_CHAT_TICKETS)
def test_decode_token_rejects_bad_chat_tickets(name):
    with pytest.raises(TokenError):
        decode_token(BAD_CHAT_TICKETS[name](), SCOPE_CHAT)


def test_token_errors_never_contain_the_token():
    secret_marker = "SECRETMARKER" + "a" * 40
    with pytest.raises(TokenError) as exc:
        decode_token(secret_marker, SCOPE_CHAT)
    assert secret_marker not in str(exc.value)


# ---------------------------- WebSocket ---------------------------------

def _connect(client, sid, token):
    return client.websocket_connect(f"/api/v1/voice/stream/{sid}?token={token}")


def test_end_to_end_real_flow_login_style_token_to_ticket_to_chat(client, fake_redis):
    """Production path only: dashboard token -> /chat-ticket -> WebSocket."""
    r = client.post("/api/v1/auth/chat-ticket", headers=bearer(dashboard_token(TENANT_A, USER_A)))
    issued = r.json()

    with ws_environment(fake_redis) as env:
        with _connect(client, issued["session_id"], issued["ticket"]) as ws:
            ws.send_json({"text": "mera refund kab aayega?"})
            reply = ws.receive_json()

    assert reply["response"] == "ok"
    assert env.captured[0]["business_id"] == TENANT_A
    assert env.captured[0]["session_id"] == issued["session_id"]
    assert any(k.endswith(f"{TENANT_A}:{issued['session_id']}") for k in fake_redis.store)


def test_dashboard_token_is_rejected_by_websocket(client, fake_redis):
    with ws_environment(fake_redis) as env, capture_logs() as logs:
        with pytest.raises(WebSocketDisconnect) as exc:
            with _connect(client, "any-session-id", dashboard_token()):
                pass
    assert exc.value.code == 1008
    assert [l["reason"] for l in logs if l["event"] == "ws_auth_rejected"] == ["InsufficientScopeError"]
    assert_untouched(env)


def test_mismatched_sid_is_rejected(client, fake_redis):
    issued = create_chat_ticket(USER_A, TENANT_A)
    other = create_chat_ticket(USER_A, TENANT_A)
    with ws_environment(fake_redis) as env, capture_logs() as logs:
        with pytest.raises(WebSocketDisconnect) as exc:
            with _connect(client, other.session_id, issued.ticket):  # valid ticket, someone else's session
                pass
    assert exc.value.code == 1008
    assert [l["reason"] for l in logs if l["event"] == "ws_auth_rejected"] == ["sid_mismatch"]
    assert_untouched(env)


def test_ticket_is_not_valid_for_another_tenants_use_of_same_session_path(client, fake_redis):
    """Tenant B holding its own ticket cannot reach tenant A's sid, and a
    ticket for B never yields A's identity."""
    a = create_chat_ticket(USER_A, TENANT_A)
    b = create_chat_ticket(uuid.uuid4(), TENANT_B)
    with ws_environment(fake_redis) as env:
        with pytest.raises(WebSocketDisconnect):
            with _connect(client, a.session_id, b.ticket):
                pass
        assert_untouched(env)


@pytest.mark.parametrize("name", BAD_CHAT_TICKETS)
def test_bad_tickets_are_rejected_by_websocket_before_any_resource_is_touched(client, fake_redis, name):
    with ws_environment(fake_redis) as env:
        with pytest.raises(WebSocketDisconnect) as exc:
            with _connect(client, "forged-sid-0123456789", BAD_CHAT_TICKETS[name]()):
                pass
        assert exc.value.code == 1008
        assert_untouched(env)


def test_missing_token_is_rejected_by_websocket(client, fake_redis):
    with ws_environment(fake_redis) as env:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/v1/voice/stream/some-session"):
                pass
        assert exc.value.code == 1008
        assert_untouched(env)


def test_client_payload_cannot_override_tenant_or_session(client, fake_redis):
    issued = create_chat_ticket(USER_A, TENANT_A)
    hostile = {
        "text": "hello", "business_id": str(TENANT_B), "tenant_id": str(TENANT_B),
        "session_id": "someone-elses-session", "sid": "someone-elses-session", "scope": "dashboard",
    }
    with ws_environment(fake_redis) as env:
        with _connect(client, issued.session_id, issued.ticket) as ws:
            ws.send_json(hostile)
            ws.receive_json()

    state = env.captured[0]
    assert state["business_id"] == TENANT_A
    assert state["session_id"] == issued.session_id
    assert all(str(TENANT_B) not in k and "someone-elses-session" not in k for k in fake_redis.store)
    env.get_or_create_call.assert_awaited_once()
    assert env.get_or_create_call.await_args.args[1:] == (TENANT_A, issued.session_id)


def test_expired_ticket_cannot_open_a_new_connection(client, fake_redis):
    just_expired = forge_token("chat", business_id=TENANT_A, sid="expired-sid-0123456789", ttl=-1)
    with ws_environment(fake_redis) as env:
        with pytest.raises(WebSocketDisconnect) as exc:
            with _connect(client, "expired-sid-0123456789", just_expired):
                pass
        assert exc.value.code == 1008
        assert_untouched(env)


# --------------------------- log hygiene --------------------------------

def test_rejected_websocket_attempts_never_log_the_credential(client, fake_redis, capsys, caplog):
    marker = "SECRETMARKER" + "b" * 40
    good = create_chat_ticket(USER_A, TENANT_A)
    with ws_environment(fake_redis), capture_logs() as logs, caplog.at_level(logging.DEBUG):
        for sid, tok in [("s1", marker), ("s2", dashboard_token()), (good.session_id + "x", good.ticket)]:
            with pytest.raises(WebSocketDisconnect):
                with _connect(client, sid, tok):
                    pass

    blob = json.dumps(logs, default=str) + caplog.text + "".join(capsys.readouterr())
    for secret in (marker, dashboard_token().split(".")[0], good.ticket):
        assert secret not in blob
    assert good.session_id not in blob      # the sid is a capability too


def test_redaction_filter_scrubs_token_query_param():
    assert redact_token("/api/v1/voice/stream/abc?token=eyJhbGciOi.payload.sig") == "/api/v1/voice/stream/abc?token=[REDACTED]"
    assert redact_token("/x?a=1&token=SECRET&b=2") == "/x?a=1&token=[REDACTED]&b=2"
    assert redact_token("/x?TOKEN=SECRET") == "/x?TOKEN=[REDACTED]"
    assert redact_token("/health") == "/health"


def test_redaction_applies_to_uvicorn_loggers_for_http_and_websocket_lines():
    install_token_redaction()
    install_token_redaction()  # idempotent
    stream = logging.StreamHandler(stream=(buf := __import__("io").StringIO()))
    loggers = [logging.getLogger("uvicorn.error"), logging.getLogger("uvicorn.access")]
    for lg in loggers:
        assert sum(isinstance(f, RedactTokenFilter) for f in lg.filters) == 1
        lg.addHandler(stream)
        lg.setLevel(logging.INFO)
    try:
        # exactly the shapes uvicorn emits
        loggers[0].info('%s - "WebSocket %s" [accepted]', ("127.0.0.1", 5000), "/api/v1/voice/stream/s?token=TOPSECRET1")
        loggers[0].info('%s - "WebSocket %s" 403', ("127.0.0.1", 5000), "/api/v1/voice/stream/s?token=TOPSECRET2")
        loggers[1].info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/x?token=TOPSECRET3", "1.1", 200)
    finally:
        for lg in loggers:
            lg.removeHandler(stream)
    out = buf.getvalue()
    assert "TOPSECRET" not in out
    assert out.count("[REDACTED]") == 3
