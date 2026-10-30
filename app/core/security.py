"""
Auth primitives: password hashing (bcrypt) and scoped JWT issuance/verification.

Two kinds of credential, deliberately NOT interchangeable:

  scope="dashboard"  Issued by /auth/login and /auth/register to a business
                     user. Long-ish lived (24h). Authorizes the REST API.
                     Never accepted by the WebSocket.

  scope="chat"       A "chat ticket", minted by an authenticated dashboard
                     user (POST /auth/chat-ticket) and handed to the customer
                     facing client. Short-lived (CHAT_TICKET_TTL_SECONDS),
                     carries no role, and is bound to ONE server-generated
                     session id (`sid`) and ONE tenant (`business_id`).
                     Authorizes only the WebSocket for that session. Never
                     accepted by REST.

Why separate them: browsers cannot set headers on a WebSocket handshake, so
the credential for the customer-facing connection ends up in a URL (query
string) where proxies and access logs can see it, and in a browser the end
customer controls. If that credential were the dashboard JWT, one leaked URL
would hand out document upload/delete and conversation history. A chat ticket
leaks far less: it expires in about a minute and can only talk to one session.

Why JWT instead of server-side sessions: any endpoint can authenticate a
request by verifying the signature, with no DB round-trip.

Why bcrypt directly instead of passlib: passlib's bcrypt backend has had
maintenance/compatibility friction with newer bcrypt releases; calling bcrypt
directly is fewer moving parts for the same guarantee.
"""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h -- dashboard session, not the customer-facing chat

SCOPE_DASHBOARD = "dashboard"
SCOPE_CHAT = "chat"

# Claims every token must carry. A token missing any of these is rejected, so
# a validly signed but incomplete token can never reach business logic.
REQUIRED_CLAIMS = ["exp", "iat", "sub", "business_id", "scope"]
_VALID_ROLES = {"admin", "agent", "viewer"}


class TokenError(Exception):
    """Token is missing, malformed, expired, badly signed or lacks required claims."""


class InsufficientScopeError(TokenError):
    """Token is authentic but is the wrong kind (e.g. a chat ticket used on REST)."""


@dataclass(frozen=True)
class TokenClaims:
    user_id: uuid.UUID
    business_id: uuid.UUID
    scope: str
    role: str | None = None  # dashboard tokens only
    sid: str | None = None   # chat tickets only


@dataclass(frozen=True)
class IssuedChatTicket:
    ticket: str
    session_id: str
    expires_in: int


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(user_id: uuid.UUID, business_id: uuid.UUID, role: str) -> str:
    """Dashboard token (scope=dashboard)."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "business_id": str(business_id),
        "role": role,
        "scope": SCOPE_DASHBOARD,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def generate_session_id() -> str:
    """256 bits from the OS CSPRNG, URL-safe (43 chars of [A-Za-z0-9_-], within
    the 50-char limit of Call.caller_ref). Unguessable, so knowing one customer's
    session id tells you nothing about anyone else's."""
    return secrets.token_urlsafe(32)


def create_chat_ticket(user_id: uuid.UUID, business_id: uuid.UUID) -> IssuedChatTicket:
    """Chat ticket (scope=chat) bound to a freshly generated session id.

    `user_id` is the dashboard user who issued it (audit trail only). The
    session id is always server-generated; callers cannot choose it.
    """
    now = datetime.now(timezone.utc)
    ttl = settings.chat_ticket_ttl_seconds
    sid = generate_session_id()
    payload = {
        "sub": str(user_id),
        "business_id": str(business_id),
        "scope": SCOPE_CHAT,
        "sid": sid,
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
    }
    ticket = jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)
    return IssuedChatTicket(ticket=ticket, session_id=sid, expires_in=ttl)


def decode_token(token: str, expected_scope: str) -> TokenClaims:
    """Verify signature, expiry, required claims and scope.

    Raises TokenError (401 / WebSocket close) for anything invalid and
    InsufficientScopeError (403) for an authentic token of the wrong kind.
    Never includes the token in an exception message.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],  # pinned: rejects alg=none and algorithm confusion
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.PyJWTError as exc:
        raise TokenError("invalid or expired token") from exc

    scope = payload["scope"]
    if scope != expected_scope:
        raise InsufficientScopeError("token scope not permitted here")

    try:
        user_id = uuid.UUID(payload["sub"])
        business_id = uuid.UUID(payload["business_id"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise TokenError("malformed identifier claim") from exc

    if scope == SCOPE_DASHBOARD:
        role = payload.get("role")
        if role not in _VALID_ROLES:
            raise TokenError("missing or invalid role claim")
        return TokenClaims(user_id=user_id, business_id=business_id, scope=scope, role=role)

    if scope == SCOPE_CHAT:
        sid = payload.get("sid")
        if not isinstance(sid, str) or not sid:
            raise TokenError("chat ticket missing session id")
        return TokenClaims(user_id=user_id, business_id=business_id, scope=scope, sid=sid)

    raise TokenError("unknown token scope")
