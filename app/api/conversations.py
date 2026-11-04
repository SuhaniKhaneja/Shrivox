"""
Conversation endpoints: REST for listing/inspecting calls, WebSocket
for the live turn-by-turn chat (text now, audio frames in Phase 2 over
the same connection contract).

Why the WebSocket takes its credential as a query param instead of a header:
Browser WebSocket clients cannot set custom Authorization headers on
the handshake request -- this is a real constraint of the WebSocket
spec, not a shortcut. The credential is therefore a short-lived *chat
ticket* (scope=chat, bound to one server-issued session id and one
tenant), never a dashboard JWT. It is verified before accept(), so no
tenant data is touched and no connection is opened for an invalid one.

Query strings end up in proxy and server access logs. Mitigations: the
ticket expires within CHAT_TICKET_TTL_SECONDS and authorizes one session
only, and app/core/log_redaction.py scrubs `token=` from uvicorn's logs.
A ticket is NOT single-use: it can be replayed until it expires.
"""

import asyncio
import hmac
import json
import time
import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.agents.graph import shrivox_agent
from app.memory.session_memory import get_history, append_turns, is_valid_session_id
from app.db.session import AsyncSessionLocal, get_db
from app.db.models import Call, Message
from app.config import settings
from app.core.failures import CATALOGUE, DB_ERRORS, REDIS_ERRORS, code_for_exception
from app.core.security import SCOPE_CHAT, TokenError, decode_token
from app.api.deps import get_current_user, CurrentUser

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["conversations"])


@router.get("/conversations")
async def list_conversations(current_user: CurrentUser = Depends(get_current_user),
                              db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Call).where(Call.business_id == current_user.business_id).order_by(Call.started_at.desc())
    )
    calls = result.scalars().all()
    return [
        {"id": str(c.id), "channel": c.channel, "status": c.status,
         "started_at": c.started_at, "ended_at": c.ended_at, "summary": c.summary}
        for c in calls
    ]


@router.get("/conversations/{call_id}")
async def get_conversation(call_id: uuid.UUID, current_user: CurrentUser = Depends(get_current_user),
                            db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Call).where(Call.id == call_id, Call.business_id == current_user.business_id))
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_result = await db.execute(select(Message).where(Message.call_id == call_id).order_by(Message.created_at))
    messages = msg_result.scalars().all()

    return {
        "id": str(call.id), "channel": call.channel, "status": call.status,
        "summary": call.summary,
        "messages": [
            {"role": m.role, "content": m.content, "intent": m.intent,
             "safety_flag": m.safety_flag, "latency_ms": m.latency_ms, "created_at": m.created_at}
            for m in messages
        ],
    }


@router.websocket("/voice/stream/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str, token: str = Query(...)):
    # Authenticate before accept()-ing so an unauthenticated caller never gets
    # a live connection to a tenant's agent, even briefly. Only a chat ticket
    # is accepted: a dashboard JWT fails the scope check. Reasons are logged by
    # category only -- never the token, and never the client-supplied session id.
    try:
        claims = decode_token(token, expected_scope=SCOPE_CHAT)
    except TokenError as exc:
        logger.warning("ws_auth_rejected", reason=type(exc).__name__)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # The ticket is bound to one session: the URL's session id must equal the
    # ticket's `sid`. Constant-time compare, since sid is a capability.
    if not hmac.compare_digest(session_id.encode(), (claims.sid or "").encode()):
        logger.warning("ws_auth_rejected", reason="sid_mismatch")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Tenant identity comes ONLY from the verified claims.
    business_id = claims.business_id

    # Defense in depth: also enforce the id format (sid is server-generated,
    # but this keeps malformed ids out of Redis keys and Call.caller_ref).
    if not is_valid_session_id(session_id):
        logger.warning("ws_auth_rejected", reason="invalid_session_id")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    async with AsyncSessionLocal() as db:
        try:
            call = await _get_or_create_call(db, business_id, session_id)
            call_id = call.id  # keep the id only: ORM attributes expire on rollback and must not be touched afterwards
        except DB_ERRORS as exc:
            await _fail(websocket, code_for_exception(exc), exc, stage="open_call")
            await _close_quietly(websocket, status.WS_1011_INTERNAL_ERROR)
            return

        outcome = "failed"
        try:
            await _serve(websocket, db, call_id, business_id, session_id)
            outcome = "disconnected"
        finally:
            await _finalize_call(db, call_id, outcome)


# ---------------------------------------------------------------- frame protocol
#
# Client -> server: a text frame holding {"text": "..."}.
# Server -> client, one frame per client message:
#   {"type": "response", "response", "intent", "escalate", "safety_flag", "latency_ms"
#    [, "degraded": ["session_memory"]]}
#   {"type": "error", "code", "message", "retryable"}      (codes: app/core/failures.py)
# An error frame does NOT close the connection; the session stays usable.

class FrameError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_client_frame(raw: str) -> str:
    """Validate one client frame and return the stripped message text."""
    if len(raw) > settings.max_message_chars * 8 + 1024:  # cheap size check before parsing
        raise FrameError("invalid_frame")
    try:
        data = json.loads(raw)
    except ValueError:
        raise FrameError("invalid_frame")
    if not isinstance(data, dict):
        raise FrameError("invalid_frame")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise FrameError("invalid_message")
    text = text.strip()
    if len(text) > settings.max_message_chars:
        raise FrameError("message_too_long")
    return text


def _error_frame(code: str) -> dict:
    message, retryable, _http = CATALOGUE[code]
    return {"type": "error", "code": code, "message": message, "retryable": retryable}


async def _send(websocket: WebSocket, payload: dict) -> bool:
    """False means the client is gone."""
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _close_quietly(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except RuntimeError:
        pass  # already closed


async def _fail(websocket: WebSocket, code: str, exc: BaseException | None, stage: str) -> bool:
    """Report a failed turn: structured log (error TYPE only, never the message,
    which can contain customer content or SQL parameters) + a typed error frame."""
    logger.error(
        "turn_failed", code=code, stage=stage,
        error_type=type(exc).__name__ if exc else "TimeoutError",
        status_code=getattr(exc, "status_code", None),
    )
    return await _send(websocket, _error_frame(code))


async def _rollback_quietly(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception as exc:  # the connection may be dead; the next operation will report it
        logger.warning("rollback_failed", error_type=type(exc).__name__)


async def _serve(websocket: WebSocket, db: AsyncSession, call_id: uuid.UUID,
                 business_id: uuid.UUID, session_id: str) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            logger.info("conversation_ended", session_id=session_id, code=message.get("code"))
            return

        raw = message.get("text")  # None for a binary frame
        try:
            if raw is None:
                raise FrameError("invalid_frame")
            user_message = parse_client_frame(raw)
        except FrameError as exc:
            if not await _send(websocket, _error_frame(exc.code)):
                return
            continue

        if not await _handle_turn(websocket, db, call_id, business_id, session_id, user_message):
            return  # the client went away mid-turn


async def _handle_turn(websocket: WebSocket, db: AsyncSession, call_id: uuid.UUID,
                       business_id: uuid.UUID, session_id: str, user_message: str) -> bool:
    """One customer message -> one frame back. Returns False if the client is gone.

    Ordering is deliberate. Postgres is the durable record, so it is written
    FIRST and a reply is only delivered once it is saved: a delivered reply is
    always persisted. Redis is working memory and is written second; if it is
    down the turn still succeeds, flagged `degraded`."""
    turn_start = time.perf_counter()
    degraded: set[str] = set()

    try:
        history = await get_history(business_id, session_id)
    except (*REDIS_ERRORS, ValueError) as exc:  # ValueError: corrupt JSON in the key
        logger.warning("redis_unavailable", op="get_history", error_type=type(exc).__name__)
        history, degraded = [], {"session_memory"}

    try:
        result = await asyncio.wait_for(
            shrivox_agent.ainvoke({
                "session_id": session_id,
                "business_id": business_id,  # from the verified ticket, never from the client
                "user_message": user_message,
                "conversation_history": history,
                "intent": None,
                "retrieved_doc_ids": [],
                "retrieved_docs": [],
                "response_text": None,
                "safety_flag": None,
                "escalate": False,
                "latency_ms": 0,
                "token_count": 0,
            }),
            timeout=settings.turn_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return await _fail(websocket, "turn_timeout", None, stage="agent")
    except Exception as exc:  # anything else: classified, logged by type, masked from the client
        return await _fail(websocket, code_for_exception(exc), exc, stage="agent")

    total_latency_ms = int((time.perf_counter() - turn_start) * 1000)

    try:
        await _persist_turn(db, call_id, len(history) // 2, user_message, result, total_latency_ms)
    except Exception as exc:
        await _rollback_quietly(db)  # leave the session usable for the next message
        return await _fail(websocket, code_for_exception(exc), exc, stage="persist")

    try:
        await append_turns(business_id, session_id, [("customer", user_message), ("agent", result["response_text"])])
    except (*REDIS_ERRORS, ValueError) as exc:
        logger.warning("redis_unavailable", op="append_turns", error_type=type(exc).__name__)
        degraded.add("session_memory")

    frame = {
        "type": "response",
        "response": result["response_text"],
        "intent": result["intent"],
        "escalate": result["escalate"],
        "safety_flag": result["safety_flag"],
        "latency_ms": total_latency_ms,
    }
    if degraded:
        frame["degraded"] = sorted(degraded)
    if result["escalate"]:
        logger.warning("conversation_escalated", session_id=session_id, business_id=str(business_id))
    return await _send(websocket, frame)


async def _finalize_call(db: AsyncSession, call_id: uuid.UUID, outcome: str) -> None:
    """Close out the call row when the connection ends. Never raises: this runs
    during teardown, where a second failure would mask the first."""
    try:
        current = (await db.execute(select(Call.status).where(Call.id == call_id))).scalar_one()
        if outcome == "failed":
            final = "failed"
        else:
            final = "escalated" if current == "escalated" else "completed"
        await db.execute(update(Call).where(Call.id == call_id).values(status=final, ended_at=datetime.utcnow()))
        await db.commit()
    except Exception as exc:  # deliberately broad and deliberately logged
        logger.error("call_finalize_failed", error_type=type(exc).__name__)
        await _rollback_quietly(db)


async def _get_or_create_call(db: AsyncSession, business_id: uuid.UUID, session_id: str) -> Call:
    result = await db.execute(
        select(Call).where(Call.business_id == business_id, Call.caller_ref == session_id)
    )
    existing = result.scalar_one_or_none()
    if existing:
        if existing.status in ("completed", "failed"):  # the customer reconnected: the call is live again
            existing.status = "active"
            existing.ended_at = None
            await db.commit()
        return existing

    call = Call(id=uuid.uuid4(), business_id=business_id, channel="browser",
                caller_ref=session_id, status="active")
    db.add(call)
    await db.commit()
    await db.refresh(call)
    return call


async def _persist_turn(db: AsyncSession, call_id: uuid.UUID, turn_index: int,
                         user_message: str, result: dict, latency_ms: int):
    customer_msg = Message(
        call_id=call_id, role="customer", content=user_message,
        intent=result["intent"],
        retrieved_doc_ids=[uuid.UUID(d) for d in result["retrieved_doc_ids"]] or None,
    )
    agent_msg = Message(
        call_id=call_id, role="agent", content=result["response_text"],
        safety_flag=result["safety_flag"], latency_ms=latency_ms,
        token_count=result["token_count"],
    )
    db.add_all([customer_msg, agent_msg])
    if result["escalate"]:
        # Same transaction as the messages: the call is marked escalated if and only if the turn is saved.
        await db.execute(update(Call).where(Call.id == call_id).values(status="escalated"))
    await db.commit()
