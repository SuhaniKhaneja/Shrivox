"""
Fixed-window rate limiting on Redis (INCR + EXPIRE). No extra dependency.

Used on login, registration, chat-ticket minting and chat messages. Keys are
`shrivox:rl:<scope>:<sha256(identifier)>` -- the identifier (an email, an IP) is
hashed so Redis never holds raw personal data.

Behaviour worth knowing:
  * Fixed windows allow a burst of up to 2x the limit across a window boundary.
    Acceptable for abuse control; this is not a billing-grade meter.
  * If Redis is unavailable the limiter FAILS OPEN and logs
    `rate_limit_unavailable`. Failing closed would lock out every customer
    whenever Redis blips; the trade-off is that brute-force protection is
    briefly off.
  * INCR and EXPIRE are separate commands. If a crash lands between them the
    counter has no expiry, so every check also repairs a missing TTL; a counter
    can never block someone forever.
"""

import hashlib
from dataclasses import dataclass

import structlog

from app.config import settings
from app.core.failures import REDIS_ERRORS, RateLimitExceeded
from app.memory import session_memory

logger = structlog.get_logger()


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int = 0


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


async def check(scope: str, identifier: str, limit: int, window_seconds: int) -> RateLimitDecision:
    if not settings.rate_limit_enabled:
        return RateLimitDecision(True)
    key = f"shrivox:rl:{scope}:{_digest(identifier)}"
    try:
        r = session_memory.get_redis()
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, window_seconds)
        ttl = await r.ttl(key)
        if ttl is None or ttl < 0:  # no expiry (crash between INCR and EXPIRE): repair it
            await r.expire(key, window_seconds)
            ttl = window_seconds
    except REDIS_ERRORS as exc:
        logger.warning("rate_limit_unavailable", scope=scope, error_type=type(exc).__name__)
        return RateLimitDecision(True)
    if count > limit:
        return RateLimitDecision(False, max(int(ttl), 1))
    return RateLimitDecision(True)


async def enforce(scope: str, identifier: str, limit: int, window_seconds: int) -> None:
    """Raise RateLimitExceeded (-> HTTP 429) when the limit is exceeded."""
    decision = await check(scope, identifier, limit, window_seconds)
    if not decision.allowed:
        logger.warning("rate_limited", scope=scope, retry_after=decision.retry_after)
        raise RateLimitExceeded(decision.retry_after)
