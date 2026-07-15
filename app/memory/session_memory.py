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
from redis.asyncio import Redis
from app.config import settings

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _key(session_id: str) -> str:
    return f"shrivox:session:{session_id}"


async def get_history(session_id: str) -> list[dict]:
    r = get_redis()
    raw = await r.get(_key(session_id))
    return json.loads(raw) if raw else []


async def append_turn(session_id: str, role: str, content: str) -> list[dict]:
    r = get_redis()
    history = await get_history(session_id)
    history.append({"role": role, "content": content})

    # Bound the window; agent layer decides whether to summarize the tail
    # it drops. We just enforce the cap here so Redis payload size and
    # prompt size stay predictable regardless of call length.
    if len(history) > settings.max_conversation_turns:
        history = history[-settings.max_conversation_turns:]

    await r.set(_key(session_id), json.dumps(history), ex=settings.session_ttl_seconds)
    return history


async def clear_session(session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(session_id))
