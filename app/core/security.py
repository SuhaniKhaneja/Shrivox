"""
Auth primitives: password hashing (bcrypt) and JWT issuance/verification.

Why JWT for this instead of server-side sessions:
The WebSocket voice endpoint and the REST dashboard endpoints are
separate connections that both need to know "which business is this."
A JWT carries business_id + user_id in the token itself, so any
endpoint can authenticate a request without a DB round-trip to look
up a session -- it decodes and verifies the signature, that's it.
Server-side sessions would need a shared session store (another Redis
key namespace) for no real benefit at this scale.

Why bcrypt directly instead of passlib:
passlib's bcrypt backend has had maintenance/compatibility friction
with newer bcrypt releases; calling bcrypt directly is fewer moving
parts for the same guarantee (adaptive hashing, per-password salt).
"""

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h -- dashboard session, not the customer-facing call


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(user_id: uuid.UUID, business_id: uuid.UUID, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "business_id": str(business_id),
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError (expired/invalid) -- caller converts to a 401."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
