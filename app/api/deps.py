"""
Shared dependencies for authenticated routes.

Why a dependency function instead of decoding the JWT inline in every
route:
Every protected endpoint needs the same three things -- verify the
token, extract business_id, reject if expired/invalid -- and every
tenant-scoped query downstream depends on getting business_id right.
Centralizing it means there's exactly one place that can get the
tenant-scoping logic wrong, not one per route handler.
"""

import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.core.security import decode_access_token

bearer_scheme = HTTPBearer()


class CurrentUser:
    def __init__(self, user_id: uuid.UUID, business_id: uuid.UUID, role: str):
        self.user_id = user_id
        self.business_id = business_id
        self.role = role


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> CurrentUser:
    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return CurrentUser(
        user_id=uuid.UUID(payload["sub"]),
        business_id=uuid.UUID(payload["business_id"]),
        role=payload["role"],
    )
