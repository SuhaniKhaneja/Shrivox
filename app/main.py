"""
FastAPI app entrypoint.

Kept thin deliberately: no business logic lives here. It wires routers
together and handles startup concerns (ensuring the pgvector extension
exists). Each router owns its own domain (auth, documents, conversations)
so route handlers stay testable in isolation from each other.
"""

import structlog
from fastapi import FastAPI
from sqlalchemy import text

from app.core.log_redaction import install_token_redaction
from app.db.session import engine
from app.api import auth, documents, conversations

logger = structlog.get_logger()
install_token_redaction()  # chat tickets travel in the WebSocket URL; keep them out of uvicorn logs
app = FastAPI(title="Shrivox AI", version="0.2.0")

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(conversations.router)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        # Idempotent -- safe on every boot. pgvector must exist before
        # any table with a Vector column can be created or queried.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("shrivox_startup_complete")


@app.get("/health")
async def health():
    return {"status": "ok"}
