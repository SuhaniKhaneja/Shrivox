"""Shared test scaffolding for the auth / tenant-isolation suites.

`forge_token` signs arbitrary claims with the real secret so tests can build
tokens the production code would never issue (expired, missing claims, odd
types). Happy-path tests should still use the production issuers
(create_access_token / POST /auth/chat-ticket) so the real flow is exercised.
"""

import time
import uuid
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import jwt

from app.config import settings


class FakeRedis:
    """Minimal in-memory stand-in so isolation is tested against the real key logic.

    Supports the session-memory list commands (RPUSH/LTRIM/EXPIRE/LRANGE) both
    directly and through a transactional pipeline. A fake cannot prove that Redis
    is atomic -- tests/integration/test_concurrency_real.py does that against a
    real server -- it only lets unit tests run the real key and window logic."""

    def __init__(self):
        self.store: dict[str, object] = {}
        self.ttls: dict[str, int] = {}
        self.executed_pipelines = 0

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)

    async def lrange(self, key, start, stop):
        items = list(self.store.get(key, []))
        return items[start:] if stop == -1 else items[start:stop + 1]

    def pipeline(self, transaction=True):
        return _FakePipeline(self)

    # synchronous primitives used by the pipeline (no awaits => atomic, like MULTI/EXEC)
    def _rpush(self, key, *values):
        self.store.setdefault(key, []).extend(values)
        return len(self.store[key])

    def _ltrim(self, key, start, stop):
        items = self.store.get(key, [])
        self.store[key] = items[start:] if stop == -1 else items[start:stop + 1]
        return True

    def _expire(self, key, seconds):
        self.ttls[key] = seconds
        return True

    def _lrange(self, key, start, stop):
        items = list(self.store.get(key, []))
        return items[start:] if stop == -1 else items[start:stop + 1]


class _FakePipeline:
    def __init__(self, redis: FakeRedis):
        self._redis, self._ops = redis, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _queue(self, name, *args):
        self._ops.append((name, args))
        return self

    def rpush(self, key, *values):
        return self._queue("_rpush", key, *values)

    def ltrim(self, key, start, stop):
        return self._queue("_ltrim", key, start, stop)

    def expire(self, key, seconds):
        return self._queue("_expire", key, seconds)

    def lrange(self, key, start, stop):
        return self._queue("_lrange", key, start, stop)

    async def execute(self):
        self._redis.executed_pipelines += 1
        return [getattr(self._redis, name)(*args) for name, args in self._ops]


def forge_token(scope="chat", *, business_id=None, user_id=None, sid="forged-sid-0123456789",
                role="admin", overrides=None, drop=(), secret=None, algorithm="HS256", ttl=60):
    now = int(time.time())
    claims = {
        "sub": str(user_id or uuid.uuid4()),
        "business_id": str(business_id or uuid.uuid4()),
        "scope": scope,
        "iat": now,
        "exp": now + ttl,
    }
    if scope == "chat":
        claims["sid"] = sid
    if scope == "dashboard":
        claims["role"] = role
    claims.update(overrides or {})
    for key in drop:
        claims.pop(key, None)
    return jwt.encode(claims, secret if secret is not None else settings.jwt_secret, algorithm=algorithm)


@contextmanager
def ws_environment(fake_redis):
    """Patch everything the WebSocket handler touches behind authentication.

    Yields a namespace whose mocks let a test assert that a REJECTED connection
    never reached the DB, the agent or Redis.
    """
    captured: list[dict] = []

    async def fake_invoke(state):
        captured.append(state)
        return {
            "response_text": "ok", "intent": "general_query", "escalate": False,
            "safety_flag": None, "retrieved_doc_ids": [], "token_count": 0,
        }

    db_ctx = MagicMock()
    db_ctx.__aenter__ = AsyncMock(return_value=MagicMock(commit=AsyncMock()))
    db_ctx.__aexit__ = AsyncMock(return_value=False)

    env = SimpleNamespace(
        captured=captured,
        redis=fake_redis,
        session_factory=MagicMock(return_value=db_ctx),
        get_or_create_call=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4(), status="active")),
        persist=AsyncMock(),
        agent=AsyncMock(side_effect=fake_invoke),
    )
    with ExitStack() as stack:
        stack.enter_context(patch("app.api.conversations.AsyncSessionLocal", env.session_factory))
        stack.enter_context(patch("app.api.conversations._get_or_create_call", env.get_or_create_call))
        stack.enter_context(patch("app.api.conversations._persist_turn", env.persist))
        stack.enter_context(patch("app.api.conversations.shrivox_agent.ainvoke", env.agent))
        stack.enter_context(patch("app.memory.session_memory.get_redis", return_value=fake_redis))
        yield env


def assert_untouched(env):
    """A rejected connection must not reach the DB, the agent, or Redis."""
    env.session_factory.assert_not_called()
    env.get_or_create_call.assert_not_called()
    env.agent.assert_not_called()
    assert env.redis.store == {}
