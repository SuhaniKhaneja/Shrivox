"""
Hot conversation memory, backed by Redis.

Why Redis and not just passing message history in the WebSocket payload:
For phone calls (Phase 3), the "client" is Twilio's media stream, not a
browser holding state in JS — there's no client-side memory to lean on.
The server must own conversation state so a call can survive a dropped
connection/reconnect, and so the LangGraph agent can be invoked
statelessly per-turn (fetch state -> run graph -> save state) without
holding Python objects in process memory, which wouldn't survive a
restart or scale past one worker.

Why TTL instead of keeping forever:
This is working memory for an in-progress call, not the permanent
record (that's Postgres, written at turn-end and call-end). If a
session goes stale (user hangs up without a clean close), we don't
want it living in Redis forever — TTL auto-expires it.

Data shape: list of {"role": "user"|"assistant", "content": str} dicts,
capped at settings.max_conversation_turns to bound prompt size (older
turns are summarized into `summary` rather than dropped silently --
summarization is added in the agent layer, not here).
"""

import json
import re
import uuid

from redis.asyncio import Redis
from app.config import settings

# Client-supplied session ids are bounded to what Call.caller_ref (String(50))
# can store and to a charset that cannot interfere with key structure.
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def is_valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _key(business_id: uuid.UUID, session_id: str) -> str:
    """Tenant boundary for hot memory: the business_id comes from the verified
    JWT, never from the client, and is part of the key. Two tenants using the
    same session_id therefore address different Redis keys."""
    return f"shrivox:session:{business_id}:{session_id}"


async def get_history(business_id: uuid.UUID, session_id: str) -> list[dict]:
    r = get_redis()
    raw = await r.get(_key(business_id, session_id))
    return json.loads(raw) if raw else []


async def append_turn(business_id: uuid.UUID, session_id: str, role: str, content: str) -> list[dict]:
    r = get_redis()
    history = await get_history(business_id, session_id)
    history.append({"role": role, "content": content})

    # Bound the window; agent layer decides whether to summarize the tail
    # it drops. We just enforce the cap here so Redis payload size and
    # prompt size stay predictable regardless of call length.
    if len(history) > settings.max_conversation_turns:
        history = history[-settings.max_conversation_turns:]

    await r.set(_key(business_id, session_id), json.dumps(history), ex=settings.session_ttl_seconds)
    return history


async def clear_session(business_id: uuid.UUID, session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(business_id, session_id))
