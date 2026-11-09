"""
Contextual retrieval against REAL PostgreSQL + pgvector.

The bug: retrieval embedded only the customer's latest message. A follow-up such as
"aur kitne din lagenge?" ("and how many days will it take?") names no topic, so its
vector matched nothing and the model answered without the policy it needed -- the
most common shape of a real support conversation.

The fix retrieves with BOTH the message alone and the message prefixed by the
customer's previous message (one batched embedding request, two tenant-scoped
searches), then keeps each chunk's best score. A topic switch still works, because
the message on its own is still searched.

HONEST SCOPE: the embedder here is a deterministic bag-of-words stand-in, so these
tests prove the MECHANISM (merge logic, tenant scoping, one embedding request).
They say nothing about how much Gemini embeddings benefit; that needs a real key
and a labelled follow-up set. The score threshold is lowered per call because
bag-of-words cosine scores are much smaller than real embedding scores.
"""

import asyncio

import pytest

from app.db.session import AsyncSessionLocal
from app.retrieval import retriever
from integration_helpers import fake_embedding, register_tenant, upload_document
from llm_fakes import ScriptedEmbedder

pytestmark = pytest.mark.integration

REFUND_DOC = "refund policy refund"
SHIPPING_DOC = "standard shipping tracking is sent by email"
PRIOR = "mera refund kab milega"
FOLLOW_UP = "aur kitne din lagenge"          # shares no word with either document
LOW = 0.1                                      # see module docstring


@pytest.fixture
def embedder(monkeypatch):
    e = ScriptedEmbedder(vector_fn=fake_embedding)
    monkeypatch.setattr("app.retrieval.retriever._genai_client", e)
    return e


@pytest.fixture
def kb(client, embedder):
    tenant = register_tenant(client, "Alpha")
    refund = upload_document(client, tenant, "refund.txt", REFUND_DOC)
    shipping = upload_document(client, tenant, "shipping.txt", SHIPPING_DOC)
    embedder.calls.clear()                      # only count the query-time requests
    return tenant, refund, shipping


def search(business_id, query, **kwargs):
    async def go():
        async with AsyncSessionLocal() as db:
            return await retriever.retrieve(db, business_id, query, score_threshold=LOW, **kwargs)
    return asyncio.run(go())


def texts(docs):
    return [d.text for d in docs]


def test_a_context_free_follow_up_retrieves_nothing_this_is_the_bug(kb):
    tenant, *_ = kb
    assert search(tenant.business_id, FOLLOW_UP) == []


def test_the_previous_message_as_context_finds_the_topic(kb):
    tenant, *_ = kb
    docs = search(tenant.business_id, FOLLOW_UP, context_query=f"{PRIOR} {FOLLOW_UP}")
    assert texts(docs) == [REFUND_DOC]


def test_a_topic_switch_still_finds_the_new_topic_first(kb):
    """The message on its own is still searched, so the context cannot drown out a
    customer who has moved on to something else."""
    tenant, *_ = kb
    message = "thanks and where is my shipping tracking"
    docs = search(tenant.business_id, message, context_query=f"{PRIOR} {message}")
    assert texts(docs)[0] == SHIPPING_DOC


def test_each_chunk_appears_once_even_if_both_queries_find_it(kb):
    tenant, *_ = kb
    docs = search(tenant.business_id, "refund policy", context_query=f"{PRIOR} refund policy")
    assert len(docs) == len({d.id for d in docs})


def test_both_queries_travel_in_ONE_embedding_request(kb, embedder):
    tenant, *_ = kb
    search(tenant.business_id, FOLLOW_UP, context_query=f"{PRIOR} {FOLLOW_UP}")
    assert embedder.calls == [[FOLLOW_UP, f"{PRIOR} {FOLLOW_UP}"]]


def test_without_context_it_is_still_a_single_embedding_of_one_text(kb, embedder):
    tenant, *_ = kb
    search(tenant.business_id, "refund policy")
    assert embedder.calls == [["refund policy"]]


def test_a_context_identical_to_the_message_is_not_embedded_twice(kb, embedder):
    tenant, *_ = kb
    search(tenant.business_id, "refund policy", context_query="refund policy")
    assert embedder.calls == [["refund policy"]]


def test_context_retrieval_never_returns_another_tenants_chunks(client, kb):
    """Another tenant holds an IDENTICAL document, so its vector is just as close.
    Both searches are filtered in SQL by business_id; neither may leak."""
    tenant, refund, _ = kb
    other = register_tenant(client, "Bravo")
    other_doc = upload_document(client, other, "refund.txt", REFUND_DOC)

    docs = search(tenant.business_id, FOLLOW_UP, context_query=f"{PRIOR} {FOLLOW_UP}")

    assert docs and all(d.metadata is not None for d in docs)
    from integration_helpers import db_rows
    owned = {r["id"] for r in db_rows("SELECT id::text AS id FROM knowledge_chunks WHERE business_id = :b",
                                      b=str(tenant.business_id))}
    assert {d.id for d in docs} <= owned
    assert other_doc["id"] not in {d.id for d in docs}


# ------------------------------------------------------- building the context query

@pytest.mark.parametrize("history,message,expected", [
    ([], "aur kitne din lagenge", None),                                           # nothing to add context from
    ([{"role": "customer", "content": "mera refund kab milega"}], "aur kitne din lagenge",
     "mera refund kab milega aur kitne din lagenge"),
    ([{"role": "customer", "content": "mera refund kab milega"}, {"role": "agent", "content": "5-7 din me aayega"}],
     "kitne din", "mera refund kab milega kitne din"),                              # the CUSTOMER's earlier message, not the agent's reply
    ([{"role": "user", "content": "refund status"}], "and when", "refund status and when"),   # legacy role name
    ([{"role": "agent", "content": "Hello! How can I help?"}], "refund?", None),    # no earlier customer message
])
def test_contextual_query_construction(history, message, expected):
    assert retriever.contextual_query(message, history) == expected


def test_a_long_message_is_assumed_self_contained():
    history = [{"role": "customer", "content": "mera refund kab milega"}]
    long_message = " ".join(["word"] * (retriever.FOLLOW_UP_MAX_WORDS + 1))
    assert retriever.contextual_query(long_message, history) is None


def test_the_borrowed_context_is_truncated():
    history = [{"role": "customer", "content": "x" * 5000}]
    query = retriever.contextual_query("and then", history)
    assert len(query) <= retriever.CONTEXT_MAX_CHARS + len(" and then")
