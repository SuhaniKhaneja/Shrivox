"""
Conversation endpoints: REST for listing/inspecting calls, WebSocket
for the live turn-by-turn chat (text now, audio frames in Phase 2 over
the same connection contract).

Why the WebSocket takes the JWT as a query param instead of a header:
Browser WebSocket clients cannot set custom Authorization headers on
the handshake request -- this is a real constraint of the WebSocket
spec, not a shortcut. Passing the token as `?token=...` and verifying
it right after `accept()` is the standard workaround. The token is
still verified before any tenant data is touched.
"""

import time
import uuid
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import jwt

from app.agents.graph import shrivox_agent
from app.memory.session_memory import get_history, append_turn
from app.db.session import AsyncSessionLocal, get_db
from app.db.models import Call, Message
from app.core.security import decode_access_token
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
    # Verify before accept()-ing so an unauthenticated caller never gets
    # a live connection to a tenant's agent, even briefly.
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    business_id = uuid.UUID(payload["business_id"])
    await websocket.accept()

    async with AsyncSessionLocal() as db:
        call = await _get_or_create_call(db, business_id, session_id)

        try:
            while True:
                incoming = await websocket.receive_json()
                user_message = incoming.get("text", "").strip()
                if not user_message:
                    continue

                turn_start = time.perf_counter()
                history = await get_history(session_id)

                result = await shrivox_agent.ainvoke({
                    "session_id": session_id,
                    "business_id": business_id,
                    "user_message": user_message,
                    "conversation_history": history,
                    "intent": None,
                    "retrieved_docs": [],
                    "retrieved_doc_ids": [],
                    "response_text": None,
                    "safety_flag": None,
                    "escalate": False,
                    "latency_ms": 0,
                    "token_count": 0,
                })

                total_latency_ms = int((time.perf_counter() - turn_start) * 1000)

                await append_turn(session_id, "customer", user_message)
                await append_turn(session_id, "agent", result["response_text"])

                await _persist_turn(db, call.id, len(history) // 2, user_message, result, total_latency_ms)

                await websocket.send_json({
                    "response": result["response_text"],
                    "intent": result["intent"],
                    "escalate": result["escalate"],
                    "safety_flag": result["safety_flag"],
                    "latency_ms": total_latency_ms,
                })

                if result["escalate"]:
                    logger.warning("conversation_escalated", session_id=session_id, business_id=str(business_id))

        except WebSocketDisconnect as e:
            call.status = "completed"
            await db.commit()
            logger.info(
                "conversation_ended",
                session_id=session_id,
                code=e.code,
                reason=getattr(e, "reason", None),
            )

async def _get_or_create_call(db: AsyncSession, business_id: uuid.UUID, session_id: str) -> Call:
    result = await db.execute(
        select(Call).where(Call.business_id == business_id, Call.caller_ref == session_id)
    )
    existing = result.scalar_one_or_none()
    if existing:
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
    await db.commit()
