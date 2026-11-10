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
from dataclasses import dataclass

import structlog

from fastapi import Depends, HTTPException, Query, status
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

logger = structlog.get_logger()


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


# ------------------------------------------------------------------- roles
#
# The role comes from the verified token (issued from the users table at login).
# Policy, in one place:
#   admin   everything, including editing the knowledge base
#   agent   read everything, and mint chat tickets (operate the chat)
#   viewer  read-only: list documents, list/read conversations
# Endpoints with no role dependency are open to any authenticated dashboard user (= read).
# A role change takes effect at the user's next login: tokens live 24h and are not revoked.

def require_role(*allowed: str):
    async def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            logger.warning("role_denied", role=user.role)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your role does not permit this action")
        return user
    return dependency


require_manager = require_role("admin")
require_operator = require_role("admin", "agent")


# -------------------------------------------------------------- pagination
#
# List endpoints used to load every row a tenant owned into memory and return it in
# one response, so cost grew without bound with history. They are now paged with a
# hard ceiling. Out-of-range values are rejected (422) rather than silently clamped,
# so a client that asks for 10,000 rows learns it is not getting them. The response
# body stays a plain JSON array; the total is returned in the X-Total-Count header.

@dataclass(frozen=True)
class Page:
    limit: int
    offset: int


MAX_OFFSET = 10_000_000


def pagination(default_limit: int = 100, max_limit: int = 500):
    """A dependency factory: Depends(pagination()) for lists, or with other bounds
    (e.g. the messages inside one conversation)."""
    def dependency(
        limit: int = Query(default_limit, ge=1, le=max_limit, description=f"Page size (1-{max_limit})."),
        offset: int = Query(0, ge=0, le=MAX_OFFSET, description="Rows to skip."),
    ) -> Page:
        return Page(limit=limit, offset=offset)
    return dependency
