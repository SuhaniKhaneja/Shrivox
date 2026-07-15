"""
Creates tables directly from SQLAlchemy metadata.

Why this instead of Alembic for Phase 1:
Alembic earns its keep once the schema changes over time and you need
versioned, reversible migrations across environments -- that's a
Phase 5 (deployment) concern. Right now the schema is new and small;
`create_all` gets you a working DB in one command. The `alembic`
dependency is already in requirements.txt for when the schema
stabilizes and migrations become worth the ceremony.

    python scripts/init_db.py
"""

import asyncio
from sqlalchemy import text
from app.db.session import engine
from app.db.models import Base


async def main():
    async with engine.begin() as conn:
        # Must exist before create_all, since KnowledgeChunk.embedding
        # is a pgvector Vector column type.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created.")


if __name__ == "__main__":
    asyncio.run(main())
