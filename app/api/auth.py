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

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Business, User
from app.core.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    business_name: str
    industry: str | None = None
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    business = Business(id=uuid.uuid4(), name=payload.business_name, industry=payload.industry)
    db.add(business)
    await db.flush()  # get business.id without committing yet

    user = User(
        id=uuid.uuid4(),
        business_id=business.id,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role="admin",
    )
    db.add(user)
    await db.commit()

    token = create_access_token(user.id, business.id, user.role)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        # Same error for "no such user" and "wrong password" -- distinguishing
        # them lets an attacker enumerate valid emails.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(user.id, user.business_id, user.role)
    return TokenResponse(access_token=token)
