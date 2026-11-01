"""Helpers for integration tests: deterministic fake embeddings, a recording
fake LLM, per-call real-Redis access, and API-driven tenant setup.

Only two things are faked: Gemini (embeddings and chat). Everything else --
the FastAPI app, auth, SQL queries, pgvector search, LangGraph, Redis -- is the
production code path.
"""

import asyncio
import hashlib
import math
import re
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from app.core.security import SCOPE_DASHBOARD, decode_token


# ---------------------------------------------------------------- embeddings

def fake_embedding(text_: str, dim: int | None = None) -> list[float]:
    """Deterministic bag-of-words embedding: identical text -> identical vector
    (cosine similarity 1.0); texts sharing words -> proportionally similar.
    Good enough to make 'another tenant has the closest vector' a real, forced
    condition rather than a hope."""
    dim = dim or settings.embedding_dim
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text_.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        vec[0], norm = 1.0, 1.0
    return [x / norm for x in vec]


async def fake_embed_text(text_: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    return fake_embedding(text_)


# ---------------------------------------------------------------- fake LLM

class FakeLLM:
    """Stands in for app.agents.graph.chat_completion and records every prompt."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def __call__(self, messages, temperature=0.3, max_tokens=300):
        self.calls.append(messages)
        is_intent_call = messages[0]["role"] == "user"
        return {
            "text": "general_query" if is_intent_call else "Here is what I can tell you.",
            "latency_ms": 1, "prompt_tokens": 1, "completion_tokens": 1,
        }

    def generation_prompts(self) -> list[str]:
        """Full text of each answer-generation prompt (system + history + user)."""
        return [
            "\n".join(m["content"] for m in call)
            for call in self.calls if call[0]["role"] == "system"
        ]


# ---------------------------------------------------------------- real Redis

class PerCallRedis:
    """Real Redis, but a fresh client per operation: redis.asyncio connections
    are bound to the event loop that created them, and every TestClient request
    runs on its own loop."""

    async def _run(self, op):
        client = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            return await op(client)
        finally:
            await client.aclose()

    async def get(self, key):
        return await self._run(lambda c: c.get(key))

    async def set(self, key, value, ex=None):
        return await self._run(lambda c: c.set(key, value, ex=ex))

    async def delete(self, key):
        return await self._run(lambda c: c.delete(key))


def redis_keys(pattern: str = "shrivox:session:*") -> list[str]:
    async def go():
        c = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            return sorted([k async for k in c.scan_iter(match=pattern)])
        finally:
            await c.aclose()
    return asyncio.run(go())


# ---------------------------------------------------------------- direct DB

def db_rows(sql: str, **params) -> list[dict]:
    """Inspect the database directly, bypassing the application, so tests assert
    on what is actually stored rather than on what the API says."""
    async def go():
        engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(sql), params)
                return [dict(r._mapping) for r in result]
        finally:
            await engine.dispose()
    return asyncio.run(go())


def retrieve_sync(business_id, query: str, top_k: int = 3):
    from app.db.session import AsyncSessionLocal
    from app.retrieval.retriever import retrieve

    async def go():
        async with AsyncSessionLocal() as db:
            return await retrieve(db, business_id, query, top_k=top_k)
    return asyncio.run(go())


# ---------------------------------------------------------------- API-driven setup

@dataclass
class Tenant:
    name: str
    email: str
    token: str            # dashboard token, from the real /auth/register
    business_id: uuid.UUID

    @property
    def auth(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}


def register_tenant(client, name: str) -> Tenant:
    email = f"{name.lower()}-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/register", json={
        "business_name": name, "email": email, "password": "correct-horse-battery",
    })
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    return Tenant(name, email, token, decode_token(token, SCOPE_DASHBOARD).business_id)


def upload_document(client, tenant: Tenant, filename: str, body: str, **extra) -> dict:
    r = client.post("/api/v1/documents/upload", headers=tenant.auth,
                    json={"filename": filename, "text": body, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def chat_once(client, tenant: Tenant, message: str) -> tuple[str, dict]:
    """Real flow: dashboard token -> /auth/chat-ticket -> WebSocket turn.
    Returns (session_id, reply_frame)."""
    r = client.post("/api/v1/auth/chat-ticket", headers=tenant.auth)
    assert r.status_code == 201, r.text
    issued = r.json()
    with client.websocket_connect(f"{issued['ws_path']}?token={issued['ticket']}") as ws:
        ws.send_json({"text": message})
        return issued["session_id"], ws.receive_json()
