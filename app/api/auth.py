"""
Auth endpoints: onboard a business + its first admin user, then log in.

Why registering a business and a user happen in one endpoint
(POST /auth/register) instead of separate "create business" then
"create user" calls:
There's no valid state where a business exists with zero users, or a
user exists with no business -- the FK is NOT NULL. Splitting this
into two calls would create a window where a client could crash
between them and leave an orphaned business row. One endpoint, one
transaction, no invalid intermediate state.
"""

import asyncio
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.ratelimit import enforce
from app.db.session import get_db
from app.db.models import Business, User
from app.api.deps import CurrentUser, require_operator
from app.core.security import hash_password, verify_password, create_access_token, create_chat_ticket, BCRYPT_MAX_BYTES

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Verified against when the email is unknown, so "no such user" costs the same
# bcrypt work as "wrong password" and response time does not reveal which emails exist.
_DUMMY_HASH = hash_password("shrivox-dummy-password-not-a-real-account")

MIN_PASSWORD_CHARS = 12


def _normalize_email(value):
    # Emails are case-insensitive in practice; storing one canonical form stops
    # "Owner@x.com" and "owner@x.com" becoming two accounts.
    return value.strip().lower() if isinstance(value, str) else value


def _client_ip(request: Request) -> str:
    # Behind a reverse proxy run uvicorn with --proxy-headers and --forwarded-allow-ips
    # so this is the real client, not the proxy.
    return request.client.host if request.client else "unknown"


class RegisterRequest(BaseModel):
    business_name: str = Field(min_length=1, max_length=200)  # Business.name is String(200)
    industry: str | None = Field(default=None, max_length=100)
    email: EmailStr
    password: str

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, value):
        return _normalize_email(value)

    @field_validator("password")
    @classmethod
    def _password_rules(cls, value: str) -> str:
        if len(value) < MIN_PASSWORD_CHARS:
            raise ValueError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
        if len(value.encode()) > BCRYPT_MAX_BYTES:
            raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes (bcrypt limit)")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)  # bounded; longer than 72 bytes simply cannot match

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, value):
        return _normalize_email(value)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatTicketResponse(BaseModel):
    ticket: str
    session_id: str
    expires_in: int  # seconds
    ws_path: str


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    await enforce("register_ip", _client_ip(request), settings.rate_limit_register_per_ip_per_hour, 3600)

    if not settings.registration_enabled:
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={
            "detail": "Self-service registration is disabled.", "code": "registration_disabled"})

    existing = await db.execute(select(User.id).where(func.lower(User.email) == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    hashed = await asyncio.to_thread(hash_password, payload.password)  # CPU-bound: keep it off the event loop

    business = Business(id=uuid.uuid4(), name=payload.business_name, industry=payload.industry)
    db.add(business)
    await db.flush()  # get business.id without committing yet

    user = User(id=uuid.uuid4(), business_id=business.id, email=payload.email, hashed_password=hashed, role="admin")
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # Lost a race with a concurrent registration of the same email. The business
        # and user are one transaction, so rolling back leaves no orphan business.
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    token = create_access_token(user.id, business.id, user.role)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Throttle BEFORE touching the database or bcrypt. Per-IP stops one client
    # spraying many accounts; per-email stops many clients hammering one account
    # (at the cost that an attacker can briefly slow a victim's logins).
    await enforce("login_ip", _client_ip(request), settings.rate_limit_login_per_ip_per_minute, 60)
    await enforce("login_email", payload.email, settings.rate_limit_login_per_email_per_15min, 900)

    result = await db.execute(select(User).where(func.lower(User.email) == payload.email))
    user = result.scalar_one_or_none()

    # Always exactly one bcrypt verification, whether or not the user exists.
    stored_hash = user.hashed_password if user else _DUMMY_HASH
    password_ok = await asyncio.to_thread(verify_password, payload.password, stored_hash)

    if not user or not password_ok:
        # Same error for "no such user" and "wrong password" -- distinguishing
        # them lets an attacker enumerate valid emails.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(user.id, user.business_id, user.role)
    return TokenResponse(access_token=token)


@router.post("/chat-ticket", response_model=ChatTicketResponse, status_code=status.HTTP_201_CREATED)
async def issue_chat_ticket(current_user: CurrentUser = Depends(require_operator)):
    """Mint a short-lived, single-session credential for the customer-facing chat.

    Requires a dashboard token with the admin or agent role (a viewer cannot start
    chats, which cost LLM calls). The request has no body on purpose: the tenant
    comes from the verified token and the session id is generated here, so
    nothing the client sends can choose either. Hand `ticket` to the customer
    client and have it open `ws_path` with `?token=<ticket>`.
    """
    await enforce("ticket_user", str(current_user.user_id), settings.rate_limit_ticket_per_user_per_minute, 60)
    issued = create_chat_ticket(current_user.user_id, current_user.business_id)
    return ChatTicketResponse(
        ticket=issued.ticket,
        session_id=issued.session_id,
        expires_in=issued.expires_in,
        ws_path=f"/api/v1/voice/stream/{issued.session_id}",
    )
