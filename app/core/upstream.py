"""
One failure policy for every call to Gemini (chat AND embeddings).

Without this, each call site would invent its own: the old chat client retried
EVERY exception three times (including a bad API key or a malformed request,
which can never succeed), had no timeout, and wrapped the final failure in an
opaque RetryError; the embedding call had no retry or timeout at all.

Policy:
  * every attempt has a timeout (UPSTREAM_TIMEOUT_SECONDS)
  * TRANSIENT failures -- timeouts, network errors, HTTP 408/429/5xx -- are
    retried, at most UPSTREAM_MAX_ATTEMPTS attempts. The wait is exponential
    backoff, EXCEPT when the server states how long to wait (Gemini's RetryInfo
    on a 429, or a Retry-After header): then that is the wait, because guessing
    shorter just burns an attempt inside the same rate-limit window. If the
    server asks for longer than UPSTREAM_MAX_RETRY_WAIT_SECONDS the call fails
    immediately instead of sleeping, and the error carries `retry_after` so the
    HTTP layer can tell the client when to come back.
  * REJECTED requests -- any other 4xx (bad request, bad key, unknown model) --
    are NOT retried: sending the same request again cannot help
  * anything else (a programming error) propagates untouched and unretried
  * cancellation always propagates immediately

Errors carry only the service name and a status code. Exception text from the
SDK is deliberately NOT copied: it can echo request content (customer
messages), so it must never reach a log line or a client.
"""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import structlog
from google.genai import errors as genai_errors
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()
T = TypeVar("T")


class UpstreamError(Exception):
    retryable = False

    def __init__(self, service: str, reason: str, status_code: int | None = None,
                 retry_after: float | None = None):
        super().__init__(f"{service}: {reason}")
        self.service = service
        self.reason = reason
        self.status_code = status_code
        self.retry_after = retry_after  # seconds the SERVER asked us to wait, when it said


class UpstreamUnavailable(UpstreamError):
    """Transient: the same request may succeed if repeated."""
    retryable = True


class UpstreamRejected(UpstreamError):
    """Permanent: the upstream refused this request; repeating it cannot help."""


class UpstreamEmptyResponse(UpstreamError):
    """The call succeeded but produced nothing usable (blocked / empty output)."""
    retryable = True


_DURATION_SECONDS = re.compile(r"^\s*(\d+(?:\.\d+)?)s?\s*$")


def _seconds(value) -> float | None:
    """'7s' / '0.5s' / '3' / 3 -> seconds; anything else (negative, nan, inf, text) -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        match = _DURATION_SECONDS.match(value)
        if not match:
            return None
        seconds = float(match.group(1))
    else:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the SERVER said to wait, or None. Reads Gemini's google.rpc.RetryInfo from the error body,
    then a Retry-After response header. Never raises: a malformed hint is simply no hint."""
    try:
        body = getattr(exc, "details", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        for item in (error.get("details") or []) if isinstance(error, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                seconds = _seconds(item.get("retryDelay"))
                if seconds is not None:
                    return seconds
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is not None:
            return _seconds(headers.get("retry-after"))
    except Exception:  # noqa: BLE001 -- a malformed hint must never turn a retryable error into a crash
        return None
    return None


def classify(exc: BaseException, service: str) -> UpstreamError | None:
    """Map an SDK/transport exception to our taxonomy, or None if it isn't ours."""
    if isinstance(exc, genai_errors.APIError):
        code = exc.code if isinstance(exc.code, int) else None
        if code is not None and (code in (408, 429) or code >= 500):
            return UpstreamUnavailable(service, f"http_{code}", code, retry_after_seconds(exc))
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
            retry_after=getattr(exc, "retry_after", None),
        )
    return before_sleep


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _worth_retrying(exc: BaseException) -> bool:
    """Transient, and not a server request to wait longer than a live call can."""
    if not isinstance(exc, UpstreamUnavailable):
        return False
    asked = exc.retry_after
    return asked is None or asked <= settings.upstream_max_retry_wait_seconds


def _wait_for(retry_state) -> float:
    """The server's stated delay when there is one, else exponential backoff."""
    asked = getattr(retry_state.outcome.exception(), "retry_after", None)
    if asked is not None:
        return asked
    return wait_exponential(multiplier=settings.upstream_retry_backoff_seconds, max=4)(retry_state)


async def call_with_retry(service: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run `fn` under the shared policy. `service` is "llm" or "embedding"."""
    retrying = AsyncRetrying(
        stop=stop_after_attempt(settings.upstream_max_attempts),
        wait=_wait_for,
        retry=retry_if_exception(_worth_retrying),
        before_sleep=_log_retry(service),
        sleep=_sleep,
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            return await _attempt(service, fn)
    raise AssertionError("unreachable")  # pragma: no cover
