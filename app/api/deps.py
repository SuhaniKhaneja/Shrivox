"""
Shared dependencies for authenticated REST routes.

Every protected REST endpoint depends on get_current_user, which accepts ONLY
a dashboard-scoped token. A chat ticket (scope=chat) is rejected here with 403
even though its signature is valid -- see app/core/security.py for why the two
credentials are kept apart.

Centralizing this means there is exactly one place that can get the
tenant-scoping logic wrong, not one per route handler.
"""

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import (
    SCOPE_DASHBOARD,
    InsufficientScopeError,
    TokenError,
    decode_token,
)

# auto_error=False so a missing/blank Authorization header yields a proper 401
# (FastAPI's default is 403, which is the wrong status for "not authenticated").
bearer_scheme = HTTPBearer(auto_error=False)

def _unauthenticated() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


class CurrentUser:
    def __init__(self, user_id: uuid.UUID, business_id: uuid.UUID, role: str):
        self.user_id = user_id
        self.business_id = business_id
        self.role = role


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    if credentials is None:
        raise _unauthenticated()

    try:
        claims = decode_token(credentials.credentials, expected_scope=SCOPE_DASHBOARD)
    except InsufficientScopeError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token scope not permitted for this endpoint")
    except TokenError:
        raise _unauthenticated()

    return CurrentUser(user_id=claims.user_id, business_id=claims.business_id, role=claims.role)
