"""
HNSW index on knowledge_chunks.embedding, and the one setting that makes it safe for many tenants.

Without an index every search scans every chunk of EVERY tenant (the tenant filter is applied
row by row), so cost grows with the whole platform, not with one tenant's knowledge base.

The catch -- why this is gated on the pgvector version: pgvector applies the tenant WHERE clause
AFTER the index scan, to the ef_search (default 40) nearest vectors. When other tenants' vectors
crowd the top 40, a small tenant gets fewer than k results, silently. pgvector >= 0.8 has
`hnsw.iterative_scan`, which keeps scanning until the filter is satisfied. So:

  * pgvector >= 0.8  -> create the index AND enable iterative scan in retrieval
  * older pgvector   -> create NO index (exact scan: slower but always complete) and warn

The index is deliberately not declared on the SQLAlchemy model: create_all would then build it
on any pgvector version, bypassing the gate. It is created here, at startup and by
scripts/init_db.py, with CREATE INDEX CONCURRENTLY so an existing production table is not
write-locked during the build.
"""

import re

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger()

ADVISORY_LOCK_KEY = 0x5348_5249_564F_58  # "SHRIVOX": serialises concurrent boots (several workers start at once)
INDEX_NAME = "ix_knowledge_chunks_embedding_hnsw"
HNSW_M = 16                # pgvector defaults; recorded here so a change is a visible diff
HNSW_EF_CONSTRUCTION = 64
MIN_ITERATIVE_VERSION = (0, 8, 0)

# Set by ensure_vector_index() at startup; read by the retriever. False (the safe default) means
# "do not issue hnsw settings", e.g. old pgvector, or the index step has not run.
ITERATIVE_SCAN_AVAILABLE = False


def supports_iterative_scan(version: str | None) -> bool:
    """Numeric comparison (0.10.0 > 0.8.0). Unknown or unparsable means NOT supported."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version or "")
    return bool(match) and tuple(int(g) for g in match.groups()) >= MIN_ITERATIVE_VERSION


async def _installed_version(conn) -> str | None:
    return (await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar()


async def ensure_vector_index(engine: AsyncEngine) -> bool:
    """Idempotent. Returns True when a usable HNSW index with iterative scan is in place.
    Never raises for "cannot create": a missing index only costs speed, so startup continues."""
    global ITERATIVE_SCAN_AVAILABLE
    ITERATIVE_SCAN_AVAILABLE = False
    try:
        # CONCURRENTLY cannot run inside a transaction block, hence AUTOCOMMIT.
        async with engine.connect() as conn:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            if not await conn.scalar(text("SELECT to_regclass('knowledge_chunks') IS NOT NULL")):
                logger.info("vector_index_skipped", reason="table_missing")  # init_db has not run yet
                return False
            # Session-level lock: a second worker waits for the first worker's build, then sees it done.
            await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": ADVISORY_LOCK_KEY})
            try:
                return await _ensure_locked(conn)
            finally:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": ADVISORY_LOCK_KEY})
    except Exception as exc:  # noqa: BLE001 -- slower search beats a failed boot; the type is logged
        logger.error("vector_index_failed", error_type=type(exc).__name__)
        ITERATIVE_SCAN_AVAILABLE = False
        return False


async def _ensure_locked(conn) -> bool:
    global ITERATIVE_SCAN_AVAILABLE
    version = await _installed_version(conn)
    if not supports_iterative_scan(version):
        logger.warning("vector_index_not_created", pgvector_version=version,
                       hint="pgvector >= 0.8 is required for a safe multi-tenant HNSW index; "
                            "searches stay exact but scan every chunk")
        return False

    state = (await conn.execute(text(
        "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE c.relname = :n"), {"n": INDEX_NAME})).scalar()
    if state is False:  # a CONCURRENTLY build that was interrupted leaves an unusable index behind
        logger.warning("vector_index_rebuilding_invalid")
        await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}"))
        state = None
    if state is None:
        logger.info("vector_index_building")
        await conn.execute(text(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON knowledge_chunks "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"))
    ITERATIVE_SCAN_AVAILABLE = True
    return True
