"""
FastAPI app entrypoint.

Kept thin deliberately: no business logic lives here. It wires routers
together and handles startup concerns (ensuring the pgvector extension
exists). Each router owns its own domain (auth, documents, conversations)
so route handlers stay testable in isolation from each other.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.failures import CATALOGUE, RateLimitExceeded, code_for_exception
from app.core.log_redaction import install_token_redaction
from app.core.observability import configure_logging, metrics_middleware, metrics_response
from app.core.upstream import UpstreamError
from app.config import settings
from app.db.session import engine
from app.api import auth, documents, conversations

logger = structlog.get_logger()
configure_logging(settings.log_level)
install_token_redaction()  # chat tickets travel in the WebSocket URL; keep them out of uvicorn logs
app = FastAPI(title="Shrivox AI", version="0.2.0")
app.middleware("http")(metrics_middleware)



async def _failure_response(request: Request, exc: Exception) -> JSONResponse:
    """Turn an infrastructure failure into a clean JSON response. Messages come
    from a fixed catalogue; the exception itself is only logged by type (SQL and
    SDK errors can embed parameters or customer content)."""
    code = code_for_exception(exc)
    message, _retryable, http_status = CATALOGUE[code]
    logger.error("request_failed", code=code, error_type=type(exc).__name__, path=request.url.path)
    retry_after = getattr(exc, "retry_after", None)
    headers = {"Retry-After": str(retry_after)} if retry_after else ({"Retry-After": "5"} if http_status == 503 else None)
    return JSONResponse(status_code=http_status, content={"detail": message, "code": code}, headers=headers)


# Registered per class so ordinary HTTPException/validation handling is untouched.
# A raw OSError in a REST handler is a refused/reset database connection.
for _exc_type in (UpstreamError, SQLAlchemyError, OSError, RateLimitExceeded):
    app.add_exception_handler(_exc_type, _failure_response)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(conversations.router)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        # Idempotent -- safe on every boot. pgvector must exist before
        # any table with a Vector column can be created or queried.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # Loud, structured warnings for settings that are fine locally but wrong in production.
    if settings.environment == "production" and settings.registration_enabled:
        logger.warning("registration_open_in_production", hint="set REGISTRATION_ENABLED=false once tenants are provisioned")
    if len(settings.jwt_secret) < 32:
        logger.warning("jwt_secret_short", length=len(settings.jwt_secret), hint="use at least 32 random characters")
    logger.info("shrivox_startup_complete", environment=settings.environment)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint. Carries no customer content (see observability.py);
    in production keep it off the public ingress and scrape it from inside the network."""
    return metrics_response()
