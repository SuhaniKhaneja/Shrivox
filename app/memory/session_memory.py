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

Data shape: a Redis LIST, one JSON object per message
({"role": "customer"|"agent", "content": str}), capped at
settings.max_conversation_turns MESSAGES (a turn is two messages) to bound
prompt size. Older messages are dropped, not summarized.

Why a LIST and not one JSON blob under one key:
The blob version did GET -> modify in Python -> SET. Two writers on the same
session (two browser tabs, or a reconnect overlapping a still-running turn --
a chat ticket is replayable until it expires) both read the same snapshot, and
the later SET silently erased the other's turn. Measured against real Redis:
29 of 30 concurrent turns were lost. A list is appended server-side:
RPUSH + LTRIM + EXPIRE + LRANGE run in one MULTI/EXEC, so concurrent turns
interleave as whole customer/agent pairs and nothing is lost. It is also O(1)
per turn instead of re-serialising the whole history every time.

Key versioning: the list lives under `shrivox:session:v2:...`. The previous
format stored a STRING under `shrivox:session:...`; reading a string key as a
list raises WRONGTYPE, and a busy session would never expire it (every turn
refreshes the TTL). A new prefix lets old keys age out untouched.
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
        # Short timeouts: a hung Redis must fail fast (the caller degrades) rather
        # than hang every chat turn behind it.
        _redis = Redis.from_url(
            settings.redis_url, decode_responses=True,
            socket_timeout=settings.redis_timeout_seconds,
            socket_connect_timeout=settings.redis_timeout_seconds,
        )
    return _redis


def _key(business_id: uuid.UUID, session_id: str) -> str:
    """Tenant boundary for hot memory: the business_id comes from the verified
    JWT, never from the client, and is part of the key. Two tenants using the
    same session_id therefore address different Redis keys."""
    return f"shrivox:session:v2:{business_id}:{session_id}"


def _legacy_key(business_id: uuid.UUID, session_id: str) -> str:
    """The pre-v2 string key. Only ever deleted, never read or written."""
    return f"shrivox:session:{business_id}:{session_id}"


# Roles that are the assistant's side of the conversation (stored as "agent").
_REPLY_ROLES = {"agent", "assistant", "model"}


def _decode(raw_items: list[str]) -> list[dict]:
    """Parse stored messages. A corrupt entry raises ValueError, which callers
    already treat like a Redis outage (degrade, don't fail the turn).

    A cap that cuts between a customer message and its reply would leave an
    orphaned reply at the head, and Gemini rejects a conversation that does not
    start with a user turn -- so leading replies are discarded."""
    history = [json.loads(item) for item in raw_items]
    while history and history[0].get("role") in _REPLY_ROLES:
        history.pop(0)
    return history


async def get_history(business_id: uuid.UUID, session_id: str) -> list[dict]:
    r = get_redis()
    return _decode(await r.lrange(_key(business_id, session_id), 0, -1))


async def append_turns(business_id: uuid.UUID, session_id: str, turns: list[tuple[str, str]]) -> list[dict]:
    """Append several (role, content) messages ATOMICALLY and return the window.

    Everything happens in one MULTI/EXEC round trip: a customer message and its
    reply are stored together or not at all (never a dangling customer message),
    and concurrent callers cannot overwrite each other (see module docstring)."""
    key = _key(business_id, session_id)
    payloads = [json.dumps({"role": role, "content": content}) for role, content in turns]
    cap = settings.max_conversation_turns

    r = get_redis()
    async with r.pipeline(transaction=True) as pipe:
        pipe.rpush(key, *payloads)
        pipe.ltrim(key, -cap, -1)  # bound the window so Redis and prompt size stay predictable
        pipe.expire(key, settings.session_ttl_seconds)
        pipe.lrange(key, 0, -1)
        results = await pipe.execute()
    return _decode(results[-1])


async def append_turn(business_id: uuid.UUID, session_id: str, role: str, content: str) -> list[dict]:
    return await append_turns(business_id, session_id, [(role, content)])


async def clear_session(business_id: uuid.UUID, session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(business_id, session_id), _legacy_key(business_id, session_id))
