"""
Authorization and abuse controls against REAL PostgreSQL + Redis (audit M1, M2, M3).

The unit tests prove the logic with fakes. These prove the parts that only exist
for real: roles read from the users table at login, the unique-email constraint
under a genuine race, legacy mixed-case rows, and the limiter counting in Redis.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.security import hash_password
from app.main import app
from integration_helpers import db_exec, db_rows, register_tenant

pytestmark = pytest.mark.integration
PASSWORD = "agent-password-123"


def add_user(business_id, email, role, password=PASSWORD):
    db_exec("INSERT INTO users (id, business_id, email, hashed_password, role, created_at) "
            "VALUES (:id, :b, :e, :h, :r, now())",
            id=str(uuid.uuid4()), b=str(business_id), e=email, h=hash_password(password), r=role)


def login(client, email, password=PASSWORD):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def headers_for(client, email):
    r = login(client, email)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def count(table):
    return db_rows(f"SELECT count(*) AS n FROM {table}")[0]["n"]


# ------------------------------------------------------------ roles, end to end

def test_roles_stored_in_the_database_decide_what_each_user_may_do(client):
    owner = register_tenant(client, "Alpha")
    add_user(owner.business_id, "agent@example.com", "agent")
    add_user(owner.business_id, "viewer@example.com", "viewer")
    agent, viewer = headers_for(client, "agent@example.com"), headers_for(client, "viewer@example.com")
    doc = {"filename": "kb.txt", "text": "Refunds are processed within seven business days."}

    # admin (from /register): everything
    uploaded = client.post("/api/v1/documents/upload", headers=owner.auth, json=doc)
    assert uploaded.status_code == 201
    doc_id = uploaded.json()["id"]

    for who, h, can_write, can_ticket in (("agent", agent, False, True), ("viewer", viewer, False, False)):
        assert client.get("/api/v1/documents", headers=h).status_code == 200, who
        assert client.get("/api/v1/conversations", headers=h).status_code == 200, who
        assert client.post("/api/v1/documents/upload", headers=h, json=doc).status_code == 403, who
        assert client.delete(f"/api/v1/documents/{doc_id}", headers=h).status_code == 403, who
        assert client.post("/api/v1/auth/chat-ticket", headers=h).status_code == (201 if can_ticket else 403), who

    assert count("documents") == 1                         # the denied writes changed nothing
    assert client.delete(f"/api/v1/documents/{doc_id}", headers=owner.auth).status_code == 200


# ------------------------------------------------------------------ accounts

def test_registering_the_same_email_in_a_different_case_is_a_conflict_and_leaves_no_orphan(client):
    register_tenant_payload = {"business_name": "Alpha", "email": "Owner@Example.com", "password": "correct-horse-battery"}
    assert client.post("/api/v1/auth/register", json=register_tenant_payload).status_code == 201
    again = client.post("/api/v1/auth/register", json={**register_tenant_payload, "business_name": "Alpha 2", "email": "OWNER@example.COM"})
    assert again.status_code == 409
    assert (count("businesses"), count("users")) == (1, 1)
    assert db_rows("SELECT email FROM users")[0]["email"] == "owner@example.com"   # stored canonical


def test_two_simultaneous_registrations_of_one_email_produce_one_account_and_one_conflict():
    """A genuine race against Postgres's unique constraint: both requests can pass
    the 'already exists?' pre-check, and exactly one must lose cleanly (409), with
    its business rolled back rather than orphaned."""
    payload = {"business_name": "Racer", "email": "race@example.com", "password": "correct-horse-battery"}

    def attempt(_):
        return TestClient(app).post("/api/v1/auth/register", json=payload).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(attempt, range(2)))
    assert statuses == [201, 409]
    assert (count("businesses"), count("users")) == (1, 1)


@pytest.mark.parametrize("attempt", ["legacy@example.com", "LEGACY@EXAMPLE.COM", "Legacy@Example.com"])
def test_a_legacy_mixed_case_row_can_still_log_in_however_the_email_is_typed(client, attempt):
    owner = register_tenant(client, "Alpha")
    add_user(owner.business_id, "Legacy@Example.com", "agent")        # stored with capitals, as older versions allowed
    assert login(client, attempt).status_code == 200


def test_wrong_password_and_unknown_email_are_indistinguishable(client):
    owner = register_tenant(client, "Alpha")
    wrong = login(client, owner.email, "not-the-password-123")
    unknown = login(client, "nobody@example.com", "not-the-password-123")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


# ---------------------------------------------------------- limits on real Redis

def rl_keys():
    from integration_helpers import redis_keys
    return redis_keys("shrivox:rl:*")


def test_login_is_throttled_in_real_redis_and_keys_hold_no_personal_data(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_login_per_ip_per_minute", 3)
    monkeypatch.setattr(settings, "rate_limit_login_per_email_per_15min", 1000)
    email = "victim@example.com"
    codes = [login(client, email, "whatever-123456").status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]

    blocked = login(client, email, "whatever-123456")
    assert blocked.json()["code"] == "rate_limited" and 1 <= int(blocked.headers["retry-after"]) <= 60
    keys = " ".join(rl_keys())
    assert "login_ip" in keys and "login_email" in keys
    assert "victim" not in keys and "example.com" not in keys and "testclient" not in keys


def test_registration_is_throttled_per_ip(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_register_per_ip_per_hour", 2)
    codes = [client.post("/api/v1/auth/register", json={
        "business_name": f"S{i}", "email": f"s{i}@example.com", "password": "correct-horse-battery"}).status_code for i in range(3)]
    assert codes == [201, 201, 429]
    assert count("businesses") == 2                         # the throttled attempt created nothing


def test_the_per_tenant_message_budget_stops_runaway_chat_and_spares_other_tenants(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_messages_per_tenant_per_minute", 2)
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")

    def url(tenant):
        issued = client.post("/api/v1/auth/chat-ticket", headers=tenant.auth).json()
        return f"{issued['ws_path']}?token={issued['ticket']}"

    def turn(ws, text="where is my order"):
        ws.send_json({"text": text})
        return ws.receive_json()

    with client.websocket_connect(url(a)) as ws:
        assert [turn(ws)["type"] for _ in range(2)] == ["response", "response"]
        limited = turn(ws)
        assert limited["type"] == "error" and limited["code"] == "rate_limited" and limited["retryable"] is True
        assert 1 <= limited["retry_after"] <= 60
        assert turn(ws)["code"] == "rate_limited"           # still over budget; the connection stays open

    with client.websocket_connect(url(b)) as ws:           # another tenant has its own budget
        assert turn(ws)["type"] == "response"

    per_tenant = {r["business_id"]: r["n"] for r in db_rows(
        "SELECT c.business_id::text AS business_id, count(*) AS n FROM messages m JOIN calls c ON c.id = m.call_id GROUP BY 1")}
    assert per_tenant[str(a.business_id)] == 4              # two turns saved; rejected messages cost nothing
    assert per_tenant[str(b.business_id)] == 2
