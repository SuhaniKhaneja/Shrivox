"""
One failure policy for every call to Gemini (chat AND embeddings).

Without this, each call site would invent its own: the old chat client retried
EVERY exception three times (including a bad API key or a malformed request,
which can never succeed), had no timeout, and wrapped the final failure in an
opaque RetryError; the embedding call had no retry or timeout at all.

Policy:
  * every attempt has a timeout (UPSTREAM_TIMEOUT_SECONDS)
  * TRANSIENT failures -- timeouts, network errors, HTTP 408/429/5xx -- are
    retried with exponential backoff, at most UPSTREAM_MAX_ATTEMPTS attempts
  * REJECTED requests -- any other 4xx (bad request, bad key, unknown model) --
    are NOT retried: sending the same request again cannot help
  * anything else (a programming error) propagates untouched and unretried
  * cancellation always propagates immediately

Errors carry only the service name and a status code. Exception text from the
SDK is deliberately NOT copied: it can echo request content (customer
messages), so it must never reach a log line or a client.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import structlog
from google.genai import errors as genai_errors
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()
T = TypeVar("T")


class UpstreamError(Exception):
    retryable = False

    def __init__(self, service: str, reason: str, status_code: int | None = None):
        super().__init__(f"{service}: {reason}")
        self.service = service
        self.reason = reason
        self.status_code = status_code


class UpstreamUnavailable(UpstreamError):
    """Transient: the same request may succeed if repeated."""
    retryable = True


class UpstreamRejected(UpstreamError):
    """Permanent: the upstream refused this request; repeating it cannot help."""


class UpstreamEmptyResponse(UpstreamError):
    """The call succeeded but produced nothing usable (blocked / empty output)."""
    retryable = True


def classify(exc: BaseException, service: str) -> UpstreamError | None:
    """Map an SDK/transport exception to our taxonomy, or None if it isn't ours."""
    if isinstance(exc, genai_errors.APIError):
        code = exc.code if isinstance(exc.code, int) else None
        if code is not None and (code in (408, 429) or code >= 500):
            return UpstreamUnavailable(service, f"http_{code}", code)
        return UpstreamRejected(service, f"http_{code}", code)
    if isinstance(exc, asyncio.TimeoutError):
        return UpstreamUnavailable(service, "timeout")
    if isinstance(exc, httpx.TransportError):
        return UpstreamUnavailable(service, "network")
    return None


async def _attempt(service: str, fn: Callable[[], Awaitable[T]]) -> T:
    try:
        return await asyncio.wait_for(fn(), timeout=settings.upstream_timeout_seconds)
    except UpstreamError:
        raise
    except Exception as exc:  # CancelledError is a BaseException and is not caught here
        mapped = classify(exc, service)
        if mapped is None:
            raise
        raise mapped from exc


def _log_retry(service: str):
    def before_sleep(retry_state):
        exc = retry_state.outcome.exception()
        cause = exc.__cause__ or exc
        logger.warning(
            "upstream_retry", service=service, attempt=retry_state.attempt_number,
            error_type=type(cause).__name__, status_code=getattr(exc, "status_code", None),
        )
    return before_sleep


async def call_with_retry(service: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run `fn` under the shared policy. `service` is "llm" or "embedding"."""
    retrying = AsyncRetrying(
        stop=stop_after_attempt(settings.upstream_max_attempts),
        wait=wait_exponential(multiplier=settings.upstream_retry_backoff_seconds, max=4),
        retry=retry_if_exception_type(UpstreamUnavailable),
        before_sleep=_log_retry(service),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            return await _attempt(service, fn)
    raise AssertionError("unreachable")  # pragma: no cover
