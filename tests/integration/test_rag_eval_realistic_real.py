"""
The realistic-corpus RAG evaluation against REAL PostgreSQL + pgvector: seeding a throwaway tenant,
retrieving through the production retriever, and deleting the tenant afterwards.

What this proves: the seeding/cleanup SQL works, the demo/other tenants are never touched or
retrieved from, cleanup happens even when the run fails, and the report has the promised shape.
HONEST SCOPE: the embedder is a deterministic bag-of-words stand-in, so the hit rates here are a
sanity check on the harness, NOT a statement about Gemini retrieval quality (that needs a real key:
`python -m app.evaluation.rag_eval --corpus realistic`).
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select

from app.db import vector_index
from app.db.models import Business, KnowledgeChunk
from app.db.session import AsyncSessionLocal, engine
from app.evaluation import rag_eval as ev
from integration_helpers import fake_embedding, register_tenant, upload_document
from llm_fakes import ScriptedEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
def embedder(monkeypatch):
    e = ScriptedEmbedder(vector_fn=fake_embedding)
    monkeypatch.setattr("app.retrieval.retriever._genai_client", e)
    monkeypatch.setattr(vector_index, "ITERATIVE_SCAN_AVAILABLE", False)   # run() builds the index; restore after
    return e


async def _counts():
    async with AsyncSessionLocal() as db:
        eval_tenants = (await db.execute(select(func.count()).select_from(Business)
                                         .where(Business.name.like("rag-eval-%")))).scalar()
        by_tenant = dict((await db.execute(select(KnowledgeChunk.business_id, func.count())
                                           .group_by(KnowledgeChunk.business_id))).all())
        return eval_tenants, by_tenant


def _run(tmp_path, **kwargs):
    async def go():
        try:
            return await ev.run(3, 0.0, tmp_path / "r.json", retrieval_only=True, limit=None,
                                corpus="realistic", **kwargs)
        finally:
            await engine.dispose()
    return asyncio.run(go())


def test_a_run_uses_a_throwaway_tenant_and_leaves_other_tenants_alone(client, embedder, tmp_path):
    bystander = register_tenant(client, "Bystander")
    upload_document(client, bystander, "refund.txt", "refund for prepaid orders is processed in 5-7 business days")
    _, before = asyncio.run(_counts())

    report = _run(tmp_path)

    eval_tenants, after = asyncio.run(_counts())
    assert eval_tenants == 0                                           # the throwaway tenant is gone
    assert after == before                                             # nobody else's chunks added, changed or removed
    assert report["meta"]["corpus"] == "realistic" and report["meta"]["corpus_chunks"] >= 55

    cases = report["cases"]
    assert all(c["error"] is None for c in cases)
    corpus_ids = {e["id"] for e in ev.load_corpus()["entries"]}
    retrieved = {i for c in cases for i in c["retrieved_ids"]}
    assert retrieved and retrieved <= corpus_ids                       # every chunk mapped to a corpus entry; none is the bystander's
    assert len(report["threshold_sweep"]) == len(ev.SWEEP_THRESHOLDS)
    assert report["summary"]["retrieval_ms_p50"] is not None and report["summary"]["by_kind"]
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["summary"] == report["summary"]
    # Sanity for the harness only (see module docstring): lexical stand-in embeddings should still find most answers.
    assert report["summary"]["hit_rate@3"] >= 0.5


def test_the_throwaway_tenant_is_deleted_even_when_the_run_fails(client, embedder, tmp_path):
    with patch.object(ev, "evaluate", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            _run(tmp_path)
    eval_tenants, by_tenant = asyncio.run(_counts())
    assert eval_tenants == 0 and by_tenant == {}
