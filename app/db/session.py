"""
Async engine + session factory, shared across the app via dependency injection.

Why a factory function and not a module-level global session:
Each request/websocket connection needs its own session — sharing one
session across concurrent requests causes transaction bleed (one
request's uncommitted changes leaking into another's queries). FastAPI's
dependency system calls `get_db()` fresh per-request and closes it after,
which is the standard safe pattern for async SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import settings

engine = create_async_engine(settings.postgres_dsn, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
