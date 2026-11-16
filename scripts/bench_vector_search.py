"""
Benchmark: tenant-scoped vector search with and without the HNSW index.

    python scripts/bench_vector_search.py --total 30000 --output bench_vector.json

Runs in a SCRATCH database (<your db>_bench, created and dropped here) using the host/credentials of
POSTGRES_DSN from .env -- it never touches your real tables. Needs CREATEDB rights.

What it measures, per tenant size class, through the real retriever._nearest():
  * latency p50/p95 (ms), database time only -- no embedding call
  * recall@k against an exact scan (ground truth for "did the index miss a neighbour")
  * rows returned vs k (a tenant getting fewer than k rows is the multi-tenant HNSW failure)
for two states: "before" (no vector index) and "after" (HNSW + iterative scan).

HONEST SCOPE: vectors are SYNTHETIC. Every tenant gets chunks around the same topic centres (every
real tenant has a refund policy), with one dominant tenant, so the data reproduces the crowding
that makes HNSW + a tenant filter hard. It says nothing about real embedding quality; retrieval
QUALITY with real embeddings is measured by app.evaluation.rag_eval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import uuid

from sqlalchemy import insert, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from app.db import vector_index
from app.db.models import Base, Business, Document, KnowledgeChunk
from app.retrieval import retriever

DIM = settings.embedding_dim
K = 3
TOPICS = 40
SIZE_CLASSES = {"small (50 chunks)": 50, "medium (500)": 500}   # plus one dominant tenant


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _near(center, spread, rng):
    return _unit([c + spread * rng.gauss(0, 1) / math.sqrt(DIM) for c in center])


async def build(session_factory, total: int, rng: random.Random):
    centers = [_unit([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(TOPICS)]
    tenants: dict[str, list[uuid.UUID]] = {name: [] for name in SIZE_CLASSES}
    plan: list[tuple[uuid.UUID, int, str]] = []
    remaining = total
    big = total // 2
    plan.append((uuid.uuid4(), big, "large"))
    remaining -= big
    tenants["large"] = [plan[0][0]]
    while remaining > 0:
        for name, size in SIZE_CLASSES.items():
            if remaining <= 0:
                break
            n = min(size, remaining)
            tid = uuid.uuid4()
            plan.append((tid, n, name))
            tenants[name].append(tid)
            remaining -= n
    async with session_factory() as db:
        for tid, _, name in plan:
            db.add(Business(id=tid, name=f"bench-{name}-{tid.hex[:6]}", industry="bench"))
        await db.flush()
        docs = {tid: uuid.uuid4() for tid, _, _ in plan}
        db.add_all(Document(id=docs[tid], business_id=tid, filename="b.txt", file_type="txt", status="ready")
                   for tid, _, _ in plan)
        await db.flush()
        for tid, n, _ in plan:
            rows = [{"id": uuid.uuid4(), "document_id": docs[tid], "business_id": tid, "content": f"c{i}",
                     "embedding": _near(centers[i % TOPICS], 0.5, rng), "chunk_metadata": {}} for i in range(n)]
            for i in range(0, len(rows), 500):
                await db.execute(insert(KnowledgeChunk.__table__), rows[i:i + 500])
        await db.commit()
    return centers, tenants


async def search(session_factory, tenant, query, exact=False):
    async with session_factory() as db:
        if exact:
            await db.execute(text("SET LOCAL enable_indexscan = off"))
            await db.execute(text("SET LOCAL enable_bitmapscan = off"))
        t0 = time.perf_counter()
        rows = await retriever._nearest(db, tenant, query, K)
        ms = (time.perf_counter() - t0) * 1000
        return [c.id for c, _ in rows], ms


async def measure(session_factory, centers, tenants, per_class: int, rng: random.Random) -> dict:
    out = {}
    for name, ids in tenants.items():
        recalls, lat, short = [], [], 0
        for tid in rng.sample(ids, min(per_class, len(ids))):
            for _ in range(3):
                query = _near(rng.choice(centers), 0.5, rng)
                truth, _ = await search(session_factory, tid, query, exact=True)
                await search(session_factory, tid, query)                      # warm-up
                got, _ = await search(session_factory, tid, query)
                times = [(await search(session_factory, tid, query))[1] for _ in range(3)]
                lat.append(statistics.median(times))
                recalls.append(len(set(got) & set(truth)) / max(len(truth), 1))
                short += len(got) < min(K, len(truth))
        lat.sort()
        out[name] = {"queries": len(lat), "recall@k": round(sum(recalls) / len(recalls), 4),
                     "p50_ms": round(lat[len(lat) // 2], 2), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2),
                     "queries_returning_fewer_than_k": short}
    return out


async def main(total: int, per_class: int, output: Path, seed: int):
    base = make_url(settings.postgres_dsn)
    bench_db = f"{base.database}_bench"
    admin = create_async_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{bench_db}"'))
        await conn.execute(text(f'CREATE DATABASE "{bench_db}"'))
    engine = create_async_engine(base.set(database=bench_db), pool_size=5)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        rng = random.Random(seed)
        t0 = time.perf_counter()
        centers, tenants = await build(factory, total, rng)
        async with engine.connect() as conn:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(text("ANALYZE"))
        load_s = time.perf_counter() - t0
        async with engine.connect() as conn:
            version = (await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))).scalar()

        vector_index.ITERATIVE_SCAN_AVAILABLE = False
        before = await measure(factory, centers, tenants, per_class, random.Random(seed + 1))

        t1 = time.perf_counter()
        assert await vector_index.ensure_vector_index(engine), "HNSW index not created (pgvector < 0.8?)"
        build_s = time.perf_counter() - t1
        async with engine.connect() as conn:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(text("ANALYZE"))
        after = await measure(factory, centers, tenants, per_class, random.Random(seed + 1))
        report = {"total_chunks": total, "tenants": sum(len(v) for v in tenants.values()), "dim": DIM, "k": K,
                  "pgvector": version, "load_seconds": round(load_s, 1), "index_build_seconds": round(build_s, 1),
                  "before_no_vector_index": before, "after_hnsw_iterative": after,
                  "note": "synthetic clustered vectors; database time only; see script docstring"}
        output.write_text(json.dumps(report, indent=2))
        print(f"{total} chunks, {report['tenants']} tenants, pgvector {version}, HNSW build {build_s:.1f}s")
        print(f"{'tenant size':<20}{'state':<8}{'recall@3':>9}{'p50 ms':>9}{'p95 ms':>9}{'<k rows':>9}")
        for name in before:
            for state, data in (("before", before), ("after", after)):
                m = data[name]
                print(f"{name:<20}{state:<8}{m['recall@k']:>9.3f}{m['p50_ms']:>9.2f}{m['p95_ms']:>9.2f}"
                      f"{m['queries_returning_fewer_than_k']:>9}")
    finally:
        vector_index.ITERATIVE_SCAN_AVAILABLE = False
        await engine.dispose()
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{bench_db}"'))
        await admin.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=30000, help="total chunks across all tenants")
    parser.add_argument("--per-class", type=int, default=6, help="tenants sampled per size class")
    parser.add_argument("--output", type=Path, default=Path("bench_vector.json"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    asyncio.run(main(args.total, args.per_class, args.output, args.seed))
