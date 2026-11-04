"""
Authorization and abuse-hardening tests (audit M1, M2 remainder, M3) -- no infrastructure.

Roles are enforced per endpoint; JWT_SECRET has a production policy; login does
the same bcrypt work whether or not the email exists; emails are case-insensitive;
passwords respect bcrypt's real limits; and a Redis-backed limiter throttles abuse.
"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import redis.exceptions
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from structlog.testing import capture_logs

from app.config import Settings, settings
from app.core.security import hash_password, verify_password
from app.db.session import get_db
from app.main import app
from auth_helpers import forge_token

GOOD_PASSWORD = "correct-horse-battery"
USER_HASH = hash_password(GOOD_PASSWORD)          # computed once: bcrypt is deliberately slow
BIZ, USER_ID = uuid.uuid4(), uuid.uuid4()


class FakeDB:
    def __init__(self, user=None, commit_error=None):
        self.user, self.commit_error = user, commit_error
        self.added, self.executes, self.rolled_back = [], 0, False

    async def execute(self, *a, **k):
        self.executes += 1
        return SimpleNamespace(scalar_one_or_none=lambda: self.user, scalars=lambda: SimpleNamespace(all=lambda: []))

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        pass

    async def commit(self):
        if self.commit_error:
            raise self.commit_error

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, obj):
        pass

    async def delete(self, obj):
        pass


@pytest.fixture
def use_db():
    def install(db):
        async def override():
            yield db
        app.dependency_overrides[get_db] = override
        return db
    yield install
    app.dependency_overrides.clear()


def existing_user():
    return SimpleNamespace(id=USER_ID, business_id=BIZ, role="admin", email="owner@example.com", hashed_password=USER_HASH)


def token(role):
    return forge_token("dashboard", business_id=BIZ, user_id=USER_ID, role=role)


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


# ============================================================ role enforcement

ENDPOINTS = {
    # name: (method, path, json, {role: expected status})   -- 404 means "authorized, row not found"
    "list_documents":    ("GET", "/api/v1/documents", None, {"admin": 200, "agent": 200, "viewer": 200}),
    "upload_document":   ("POST", "/api/v1/documents/upload", {"filename": "a.txt", "text": "hello"}, {"admin": 201, "agent": 403, "viewer": 403}),
    "delete_document":   ("DELETE", f"/api/v1/documents/{uuid.uuid4()}", None, {"admin": 404, "agent": 403, "viewer": 403}),
    "list_conversations": ("GET", "/api/v1/conversations", None, {"admin": 200, "agent": 200, "viewer": 200}),
    "get_conversation":  ("GET", f"/api/v1/conversations/{uuid.uuid4()}", None, {"admin": 404, "agent": 404, "viewer": 404}),
    "issue_chat_ticket": ("POST", "/api/v1/auth/chat-ticket", None, {"admin": 201, "agent": 201, "viewer": 403}),
}
MATRIX = [(name, role, status) for name, (_, _, _, by_role) in ENDPOINTS.items() for role, status in by_role.items()]


@pytest.mark.parametrize("name,role,expected", MATRIX, ids=[f"{n}-{r}" for n, r, _ in MATRIX])
def test_role_permission_matrix(use_db, name, role, expected):
    use_db(FakeDB())
    method, path, body, _ = ENDPOINTS[name]
    with patch("app.api.documents.embed_texts", new=AsyncMock(side_effect=lambda texts, task_type=None: [[0.0] * 3 for _ in texts])):
        r = TestClient(app, raise_server_exceptions=False).request(method, path, json=body, headers=bearer(token(role)))
    assert r.status_code == expected, r.text


def test_every_dashboard_endpoint_is_covered_by_the_role_matrix():
    """A new route that forgets role handling must not slip through unnoticed."""
    covered = {(m, p.split("/")[-1] if "/api/v1/documents/" in p or "/conversations/" in p else p) for m, p, _, _ in ENDPOINTS.values()}
    declared = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/api/v1/") and not path.startswith("/api/v1/voice") and path not in (
                "/api/v1/auth/login", "/api/v1/auth/register"):
            for m in getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}:
                declared.add((m, path))
    assert len(declared) == len(ENDPOINTS), f"routes changed; update ENDPOINTS: {sorted(declared)}"


# ============================================================ JWT_SECRET policy

def build(**kw):
    return Settings(_env_file=None, google_api_key="k", **kw)


GOOD_SECRET = "9fK2vQ8xLm4Zr7TnB3cYw6HdJ1pXs5Ae"


# Each case trips exactly ONE rule, so removing any single rule makes a test fail.
VARIED_31 = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1f"          # 31 chars, plenty of variety, no placeholder text
VARIED_32 = VARIED_31 + "g"


@pytest.mark.parametrize("secret,rule", [
    ("Kx9fQ2mZ7pLw", "too short only"),
    (VARIED_31, "one character under the minimum"),
    ("change_me_" + "Qx9fK2mZ7pLwR4tY8uN3vB6c", "placeholder only (long and varied)"),
    ("Replace_Me" + "-Qx9fK2mZ7pLwR4tY8uN3vB6", "placeholder only (another marker)"),
    ("ab" * 20, "too little variety only (long, no placeholder)"),
    ("a" * 40, "too little variety only"),
])
def test_production_rejects_each_weak_secret_rule_independently(secret, rule):
    assert len(VARIED_31) == 31 and len(set(VARIED_31)) >= 8
    with pytest.raises(ValidationError):
        build(jwt_secret=secret, environment="production")


def test_production_accepts_a_secret_exactly_at_the_minimum_length():
    assert len(VARIED_32) == 32
    assert build(jwt_secret=VARIED_32, environment="production").environment == "production"


def test_production_accepts_a_strong_secret():
    assert build(jwt_secret=GOOD_SECRET, environment="production").environment == "production"


def test_development_keeps_short_secrets_working_but_never_empty():
    assert build(jwt_secret="ci_test_secret").environment == "development"
    with pytest.raises(ValidationError):
        build(jwt_secret="   ")


def test_environment_must_be_a_known_value():
    with pytest.raises(ValidationError):
        build(jwt_secret=GOOD_SECRET, environment="prod")


# ======================================================= password / input rules

@pytest.mark.parametrize("password,why", [
    ("short1", "under 12 characters"),
    ("x" * 73, "over bcrypt's 72-byte limit"),
    ("😀" * 25, "25 characters but 100 bytes"),
])
def test_registration_rejects_unusable_passwords(use_db, password, why):
    db = use_db(FakeDB())
    r = TestClient(app).post("/api/v1/auth/register", json={"business_name": "Shop", "email": "a@example.com", "password": password})
    assert r.status_code == 422, why
    assert db.executes == 0 and not db.added


@pytest.mark.parametrize("payload", [
    {"business_name": "", "email": "a@example.com", "password": GOOD_PASSWORD},
    {"business_name": "x" * 201, "email": "a@example.com", "password": GOOD_PASSWORD},
    {"business_name": "Shop", "industry": "i" * 101, "email": "a@example.com", "password": GOOD_PASSWORD},
    {"business_name": "Shop", "email": "not-an-email", "password": GOOD_PASSWORD},
])
def test_registration_rejects_oversized_or_invalid_fields(use_db, payload):
    use_db(FakeDB())
    assert TestClient(app).post("/api/v1/auth/register", json=payload).status_code == 422


def test_password_of_exactly_72_bytes_is_accepted_and_73_is_not_hashable():
    assert verify_password("p" * 72, hash_password("p" * 72))
    with pytest.raises(ValueError):
        hash_password("p" * 73)


def test_verifying_an_overlong_password_is_false_not_an_error():
    assert verify_password("p" * 200, USER_HASH) is False
    assert verify_password("anything", "not-a-bcrypt-hash") is False


# ============================================================ login behaviour

@pytest.fixture
def verify_spy(monkeypatch):
    calls = []
    real = verify_password

    def spy(plain, hashed):
        calls.append(hashed)
        return real(plain, hashed)
    monkeypatch.setattr("app.api.auth.verify_password", spy)
    return calls


def login(email, password):
    return TestClient(app).post("/api/v1/auth/login", json={"email": email, "password": password})


def test_unknown_email_costs_the_same_bcrypt_work_as_a_wrong_password(use_db, verify_spy):
    from app.api import auth
    use_db(FakeDB(user=None))
    unknown = login("nobody@example.com", GOOD_PASSWORD)
    assert verify_spy == [auth._DUMMY_HASH]               # a hash was verified even though no user exists

    verify_spy.clear()
    use_db(FakeDB(user=existing_user()))
    wrong = login("owner@example.com", "wrong-password-123")
    assert verify_spy == [USER_HASH]

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()                 # indistinguishable responses


def test_correct_credentials_log_in_and_the_token_carries_the_stored_role(use_db):
    from app.core.security import SCOPE_DASHBOARD, decode_token
    user = existing_user()
    user.role = "agent"
    use_db(FakeDB(user=user))
    r = login("owner@example.com", GOOD_PASSWORD)
    assert r.status_code == 200
    assert decode_token(r.json()["access_token"], SCOPE_DASHBOARD).role == "agent"


def test_overlong_login_password_is_a_clean_401(use_db):
    use_db(FakeDB(user=existing_user()))
    assert login("owner@example.com", "p" * 500).status_code == 401


# ======================================================= registration behaviour

def test_registration_normalizes_the_email_and_creates_an_admin(use_db):
    db = use_db(FakeDB())
    r = TestClient(app).post("/api/v1/auth/register", json={"business_name": "Shop", "email": "  Owner@Example.COM ", "password": GOOD_PASSWORD})
    assert r.status_code == 201
    user = next(o for o in db.added if hasattr(o, "hashed_password"))
    assert user.email == "owner@example.com" and user.role == "admin"


def test_duplicate_email_is_a_409_and_creates_nothing(use_db):
    db = use_db(FakeDB(user=existing_user()))
    r = TestClient(app).post("/api/v1/auth/register", json={"business_name": "Shop", "email": "owner@example.com", "password": GOOD_PASSWORD})
    assert r.status_code == 409 and not db.added


def test_losing_an_insert_race_is_a_409_with_rollback_not_a_500(use_db):
    db = use_db(FakeDB(commit_error=IntegrityError("INSERT users", {}, Exception("duplicate key value"))))
    r = TestClient(app, raise_server_exceptions=False).post(
        "/api/v1/auth/register", json={"business_name": "Shop", "email": "race@example.com", "password": GOOD_PASSWORD})
    assert r.status_code == 409 and db.rolled_back
    assert "duplicate key" not in r.text


def test_registration_can_be_disabled(use_db, monkeypatch):
    monkeypatch.setattr(settings, "registration_enabled", False)
    db = use_db(FakeDB())
    r = TestClient(app).post("/api/v1/auth/register", json={"business_name": "Shop", "email": "a@example.com", "password": GOOD_PASSWORD})
    assert r.status_code == 403 and r.json()["code"] == "registration_disabled"
    assert db.executes == 0 and not db.added


# ============================================================== rate limiter

class RLRedis:
    def __init__(self):
        self.counts, self.ttls, self.calls, self.fail = {}, {}, [], False

    async def incr(self, key):
        if self.fail:
            raise redis.exceptions.ConnectionError("down")
        self.calls.append(("incr", key))
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key, seconds):
        self.calls.append(("expire", key, seconds))
        self.ttls[key] = seconds
        return True

    async def ttl(self, key):
        return self.ttls.get(key, -1)

    def expire_window(self, key):
        self.counts.pop(key, None)
        self.ttls.pop(key, None)


@pytest.fixture
def rl_redis(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    r = RLRedis()
    with patch("app.memory.session_memory.get_redis", return_value=r):
        yield r


def check(scope, ident, limit, window):
    from app.core.ratelimit import check as _check
    return asyncio.run(_check(scope, ident, limit, window))


@pytest.mark.ratelimit
def test_limiter_allows_up_to_the_limit_then_blocks_with_a_retry_after(rl_redis):
    assert all(check("login_ip", "1.2.3.4", 3, 60).allowed for _ in range(3))
    blocked = check("login_ip", "1.2.3.4", 3, 60)
    assert blocked.allowed is False and blocked.retry_after == 60


@pytest.mark.ratelimit
def test_limiter_counts_per_identifier_and_per_scope(rl_redis):
    for _ in range(3):
        check("login_ip", "1.2.3.4", 3, 60)
    assert check("login_ip", "1.2.3.4", 3, 60).allowed is False
    assert check("login_ip", "5.6.7.8", 3, 60).allowed is True        # another client
    assert check("register_ip", "1.2.3.4", 3, 60).allowed is True     # another action


@pytest.mark.ratelimit
def test_limiter_window_resets(rl_redis):
    for _ in range(4):
        check("s", "x", 3, 60)
    key = next(k for _, k, *_ in rl_redis.calls)
    rl_redis.expire_window(key)                                        # the window elapsed
    assert check("s", "x", 3, 60).allowed is True


@pytest.mark.ratelimit
def test_limiter_keys_never_contain_the_raw_identifier(rl_redis):
    check("login_email", "victim@example.com", 3, 60)
    check("login_ip", "203.0.113.9", 3, 60)
    keys = " ".join(k for _, k, *_ in rl_redis.calls)
    assert "victim" not in keys and "example.com" not in keys and "203.0.113.9" not in keys
    assert "login_email" in keys and keys.count("shrivox:rl:") >= 2


@pytest.mark.ratelimit
def test_a_counter_left_without_expiry_heals_itself(rl_redis):
    rl_redis.counts["shrivox:rl:s:" + __import__("app.core.ratelimit", fromlist=["x"])._digest("x")] = 5   # e.g. crash between INCR and EXPIRE
    check("s", "x", 100, 60)
    assert any(c[0] == "expire" for c in rl_redis.calls)


@pytest.mark.ratelimit
def test_limiter_fails_open_and_logs_when_redis_is_down(rl_redis):
    rl_redis.fail = True
    with capture_logs() as logs:
        decision = check("login_ip", "1.2.3.4", 1, 60)
    assert decision.allowed is True
    assert [l["error_type"] for l in logs if l["event"] == "rate_limit_unavailable"] == ["ConnectionError"]


def test_limiter_does_nothing_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    with patch("app.memory.session_memory.get_redis", side_effect=AssertionError("redis must not be touched")):
        assert check("login_ip", "1.2.3.4", 1, 60).allowed is True


@pytest.mark.ratelimit
def test_login_endpoint_returns_429_with_retry_after_and_skips_the_database(use_db, rl_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_login_per_ip_per_minute", 3)
    monkeypatch.setattr(settings, "rate_limit_login_per_email_per_15min", 100)
    db = use_db(FakeDB(user=None))
    client = TestClient(app)
    codes = [client.post("/api/v1/auth/login", json={"email": f"u{i}@example.com", "password": "whatever-123456"}).status_code for i in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    blocked = client.post("/api/v1/auth/login", json={"email": "u9@example.com", "password": "whatever-123456"})
    assert blocked.json()["code"] == "rate_limited" and int(blocked.headers["retry-after"]) >= 1
    assert db.executes == 3                                            # blocked requests never touched the database


@pytest.mark.ratelimit
def test_login_is_also_limited_per_email_across_different_ips(use_db, rl_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_login_per_ip_per_minute", 1000)
    monkeypatch.setattr(settings, "rate_limit_login_per_email_per_15min", 2)
    use_db(FakeDB(user=None))
    client = TestClient(app)
    body = {"email": "victim@example.com", "password": "whatever-123456"}
    assert [client.post("/api/v1/auth/login", json=body).status_code for _ in range(3)] == [401, 401, 429]
    other = client.post("/api/v1/auth/login", json={**body, "email": "someone-else@example.com"})
    assert other.status_code == 401


@pytest.mark.ratelimit
def test_registration_is_limited_per_ip(use_db, rl_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_register_per_ip_per_hour", 2)
    use_db(FakeDB())
    client = TestClient(app)
    codes = [client.post("/api/v1/auth/register", json={"business_name": f"S{i}", "email": f"s{i}@example.com", "password": GOOD_PASSWORD}).status_code for i in range(3)]
    assert codes == [201, 201, 429]


@pytest.mark.ratelimit
def test_ticket_minting_is_limited_per_user(use_db, rl_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_ticket_per_user_per_minute", 2)
    use_db(FakeDB())
    client = TestClient(app)
    h = bearer(token("admin"))
    assert [client.post("/api/v1/auth/chat-ticket", headers=h).status_code for _ in range(3)] == [201, 201, 429]
    other_user = bearer(forge_token("dashboard", business_id=BIZ, user_id=uuid.uuid4(), role="admin"))
    assert client.post("/api/v1/auth/chat-ticket", headers=other_user).status_code == 201


@pytest.mark.parametrize("field,bad", [
    ("rate_limit_login_per_ip_per_minute", 0), ("rate_limit_register_per_ip_per_hour", 0),
    ("rate_limit_messages_per_tenant_per_minute", 0), ("rate_limit_ticket_per_user_per_minute", 100001),
])
def test_rate_limit_settings_are_bounded(field, bad):
    with pytest.raises(ValidationError):
        build(jwt_secret="s", **{field: bad})
