"""
The single catalogue of failure codes the application can report.

Every code has one fixed, customer-safe message and a `retryable` flag meaning
"the SAME request may succeed if sent again unchanged". Messages are fixed
strings on purpose: they are never built from exception text, because SQLAlchemy
and SDK errors can embed SQL, parameters or customer content.

Used by the WebSocket handler (error frames) and by the REST exception
handlers in app/main.py, so both channels describe the same failure the same way.
"""

import redis.exceptions
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError

from app.core.upstream import UpstreamEmptyResponse, UpstreamError, UpstreamRejected, UpstreamUnavailable

# code -> (customer-safe message, retryable, HTTP status when raised from a REST endpoint)
CATALOGUE: dict[str, tuple[str, bool, int]] = {
    "invalid_frame": ('Send a text frame containing a JSON object such as {"text": "..."}.', False, 400),
    "invalid_message": ('The "text" field must be a non-empty string.', False, 422),
    "message_too_long": ("The message is longer than the allowed limit.", False, 413),
    "llm_unavailable": ("The assistant is temporarily unavailable. Please try again.", True, 503),
    "llm_rejected": ("The assistant could not process this request.", False, 502),
    "llm_empty_response": ("The assistant returned no answer. Please try again or rephrase.", True, 502),
    "embedding_unavailable": ("Knowledge search is temporarily unavailable. Please try again.", True, 503),
    "embedding_rejected": ("Knowledge search could not process this request.", False, 502),
    "database_unavailable": ("The service is temporarily unavailable. Please try again.", True, 503),
    "turn_timeout": ("The request took too long. Please try again.", True, 504),
    "internal_error": ("Something went wrong. Please try again.", False, 500),
}

# Exceptions that mean "the database (or the network path to it) failed".
# A raw OSError is included because a refused/reset connection surfaces from
# the async driver before SQLAlchemy can wrap it.
DB_ERRORS = (SQLAlchemyError, OSError)
REDIS_ERRORS = (redis.exceptions.RedisError, OSError)


def is_db_unavailable(exc: BaseException) -> bool:
    """Connectivity/availability failures, as opposed to the database rejecting
    our data (a bug or bad input), which is an internal error."""
    return isinstance(exc, (OperationalError, InterfaceError, OSError)) or bool(getattr(exc, "connection_invalidated", False))


def code_for_exception(exc: BaseException) -> str:
    if isinstance(exc, UpstreamUnavailable):
        return f"{exc.service}_unavailable"
    if isinstance(exc, UpstreamRejected):
        return f"{exc.service}_rejected"
    if isinstance(exc, UpstreamEmptyResponse):
        return "llm_empty_response" if exc.service == "llm" else f"{exc.service}_unavailable"
    if isinstance(exc, UpstreamError):
        return "internal_error"
    if isinstance(exc, DB_ERRORS) and is_db_unavailable(exc):
        return "database_unavailable"
    return "internal_error"
