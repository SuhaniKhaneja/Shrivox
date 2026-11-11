"""
The multi-tenant trap with an HNSW index, against REAL PostgreSQL + pgvector.

pgvector applies a WHERE clause AFTER the index scan. With ef_search=40 the index hands back
the 40 globally-nearest vectors and the tenant filter is applied to those. Every tenant has a
refund policy, so when a big tenant's refund chunks are nearer to the query than a small
tenant's, all 40 candidates can belong to the big tenant: the small tenant gets FEWER than k
results (even none) although it has the answer. Silent, and it only appears at scale.

pgvector >= 0.8 fixes this with iterative scans (keep scanning until the filter is satisfied).
app/db/vector_index.py therefore (a) only creates the index where that is available and
(b) turns it on for retrieval. These tests build the bad case and prove the result is complete,
exact and tenant-scoped.

HONEST SCOPE: vectors are synthetic (a tight cluster for the big tenants, a looser one for the
small tenant). That reproduces the mechanism, not real embedding geometry.
"""

import asyncio
import math
import random
import uuid

import pytest
from sqlalchemy import text

from app.config import settings
from app.db import vector_index
from app.db.models import Business, Document, KnowledgeChunk
from app.db.session import AsyncSessionLocal
from app.retrieval import retriever

pytestmark = pytest.mark.integration

DIM = settings.embedding_dim
BIG_TENANTS, CHUNKS_PER_BIG, SMALL_CHUNKS = 4, 500, 30
K = 3


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _around(center, spread, rng):
    return _unit([c + spread * rng.gauss(0, 1) / math.sqrt(DIM) for c in center])


async def _seed():
    rng = random.Random(7)
    center = _unit([rng.gauss(0, 1) for _ in range(DIM)])
    async with AsyncSessionLocal() as db:
        async def tenant(name, n, spread):
            b = Business(id=uuid.uuid4(), name=name, industry="retail")
            d = Document(id=uuid.uuid4(), business_id=b.id, filename="kb.txt", file_type="txt", status="ready")
            db.add_all([b, d])
            await db.flush()
            db.add_all(KnowledgeChunk(document_id=d.id, business_id=b.id, content=f"{name} chunk {i}",
                                      embedding=_around(center, spread, rng), chunk_metadata={})
                       for i in range(n))
            return b.id
        big = [await tenant(f"big{i}", CHUNKS_PER_BIG, 0.15) for i in range(BIG_TENANTS)]
        small = await tenant("small", SMALL_CHUNKS, 1.5)   # same topic, but looser: ranks below every big chunk
        await db.commit()
    return big, small, center


async def _search_forced_hnsw(business_id, query, iterative: bool):
    """_nearest() on a plan the planner would choose for a large unselective table: the HNSW
    index. (The btree on business_id is dropped INSIDE a transaction that is rolled back, which
    stands in for 'planner picks HNSW'; the assertion on EXPLAIN keeps that honest.)"""
    async with AsyncSessionLocal() as db:
        await db.execute(text("DROP INDEX IF EXISTS ix_knowledge_chunks_business_id"))
        await db.execute(text("SET LOCAL enable_seqscan = off"))
        vector_index.ITERATIVE_SCAN_AVAILABLE = iterative
        try:
            vec = "[" + ",".join(map(str, query)) + "]"
            plan = (await db.execute(text(
                "EXPLAIN SELECT id FROM knowledge_chunks WHERE business_id = :b "
                f"ORDER BY embedding <=> '{vec}'::vector LIMIT {K}"), {"b": business_id})).scalars().all()
            rows = [(c.id, c.business_id) for c, _ in await retriever._nearest(db, business_id, query, K)]
        finally:
            vector_index.ITERATIVE_SCAN_AVAILABLE = False
            await db.rollback()
    return "\n".join(plan), rows


@pytest.fixture
def seeded():
    big, small, center = asyncio.run(_seed())
    assert asyncio.run(vector_index.ensure_vector_index(AsyncSessionLocal.kw["bind"])) is True
    asyncio.run(_analyze())
    return big, small, center


async def _analyze():
    async with AsyncSessionLocal() as db:
        await db.execute(text("ANALYZE knowledge_chunks"))
        await db.commit()


def test_without_iterative_scan_the_small_tenant_is_starved(seeded):
    """Documents the problem the fix exists for. If this ever stops reproducing (a newer
    pgvector changes defaults), the fix is no longer needed and this test says so."""
    _, small, center = seeded
    plan, rows = asyncio.run(_search_forced_hnsw(small, center, iterative=False))
    assert "hnsw" in plan.lower(), plan
    assert len(rows) < K, f"expected starvation, got {len(rows)} rows"


def test_with_iterative_scan_the_small_tenant_gets_full_exact_tenant_scoped_results(seeded):
    _, small, center = seeded
    plan, rows = asyncio.run(_search_forced_hnsw(small, center, iterative=True))
    assert "hnsw" in plan.lower(), plan
    assert len(rows) == K

    async def exact():
        async with AsyncSessionLocal() as db:
            await db.execute(text("SET LOCAL enable_indexscan = off"))
            await db.execute(text("SET LOCAL enable_bitmapscan = off"))
            return [(c.id, c.business_id) for c, _ in await retriever._nearest(db, small, center, K)]
    truth = asyncio.run(exact())
    assert [i for i, _ in rows] == [i for i, _ in truth]                     # same chunks, same order
    assert all(b == small for _, b in rows)                      # never another tenant's chunk


def test_a_big_tenant_is_unaffected(seeded):
    big, _, center = seeded
    _, rows = asyncio.run(_search_forced_hnsw(big[0], center, iterative=True))
    assert len(rows) == K and all(b == big[0] for _, b in rows)


def test_ensure_vector_index_is_idempotent_and_leaves_a_valid_index(seeded):
    bind = AsyncSessionLocal.kw["bind"]
    assert asyncio.run(vector_index.ensure_vector_index(bind)) is True       # second call: no error, nothing to do

    async def valid():
        async with AsyncSessionLocal() as db:
            return (await db.execute(text(
                "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE c.relname = :n"), {"n": vector_index.INDEX_NAME})).scalar()
    assert asyncio.run(valid()) is True


def test_old_pgvector_gets_no_index_rather_than_a_silently_incomplete_one(monkeypatch):
    async def go():
        engine = AsyncSessionLocal.kw["bind"]
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP INDEX IF EXISTS {vector_index.INDEX_NAME}"))
        monkeypatch.setattr(vector_index, "_installed_version", lambda conn: _const("0.6.0"))
        assert await vector_index.ensure_vector_index(engine) is False
        async with engine.connect() as conn:
            return (await conn.execute(text("SELECT to_regclass(:n)"), {"n": vector_index.INDEX_NAME})).scalar()
    assert asyncio.run(go()) is None
    assert vector_index.ITERATIVE_SCAN_AVAILABLE is False


async def _const(value):
    return value
