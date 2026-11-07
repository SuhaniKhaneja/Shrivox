"""
Tenant isolation at the SQL / pgvector layer, against real PostgreSQL + Redis.

The Step 1-2 suites prove isolation at the token, Redis-key and handler layers
using mocks. These tests prove it where the data actually lives: every query
site that touches tenant data, exercised through the real application.

Method: two tenants are created through the public API (/auth/register), given
IDENTICAL knowledge, and driven through the real flows. Identical content is
deliberate -- it forces "the other tenant's row is equally close" to be true,
so a missing business_id filter cannot hide behind a weaker match. Every
negative assertion has a positive control proving the data is reachable by its
owner, so none of these can pass vacuously.

Query sites covered (mutation-checked, see the PR description):
  retriever.retrieve                 documents.list_documents
  documents.delete_document          conversations.list_conversations
  conversations.get_conversation     conversations._get_or_create_call
  session_memory._key (real Redis)
"""

import uuid

import pytest

from auth_helpers import forge_token
from integration_helpers import (
    chat_once, db_rows, redis_keys, register_tenant, retrieve_sync, upload_document,
)

pytestmark = pytest.mark.integration

SHARED_POLICY = "Refunds for damaged items are processed within seven business days after the returned item is inspected."
SHARED_QUERY = "how long do refunds for damaged items take after inspection"
SECRET = "Confidential wholesale margin for ZEBRA QUOKKA program is forty one percent do not share"
UNRELATED = "Our store opens at nine every morning and closes at six in the evening on weekdays"


def chunk_ids(business_id) -> set[str]:
    return {r["id"] for r in db_rows("SELECT id::text AS id FROM knowledge_chunks WHERE business_id = :b", b=str(business_id))}


# ------------------------------------------------------------------ retrieval

def test_retrieval_returns_only_the_callers_chunks_when_the_other_tenant_has_identical_text(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    upload_document(client, a, "refunds.txt", SHARED_POLICY)
    upload_document(client, b, "refunds.txt", SHARED_POLICY)
    a_ids, b_ids = chunk_ids(a.business_id), chunk_ids(b.business_id)
    assert a_ids and b_ids and not (a_ids & b_ids)

    for tenant, own, other in ((a, a_ids, b_ids), (b, b_ids, a_ids)):
        found = {d.id for d in retrieve_sync(tenant.business_id, SHARED_QUERY, top_k=5)}
        assert found, "positive control: the owner must be able to retrieve its own chunk"
        assert found <= own
        assert not (found & other)


def test_retrieval_never_surfaces_another_tenants_confidential_text(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    upload_document(client, a, "hours.txt", UNRELATED)
    upload_document(client, b, "margins.txt", SECRET)

    owner_hits = retrieve_sync(b.business_id, SECRET)
    assert [d.text for d in owner_hits] == [SECRET], "positive control: owner retrieves its secret"

    attacker_hits = retrieve_sync(a.business_id, SECRET, top_k=10)
    assert all("ZEBRA" not in d.text for d in attacker_hits)
    assert {d.id for d in attacker_hits} <= chunk_ids(a.business_id)


# -------------------------------------------------------------------- documents

def test_document_list_is_scoped_to_the_authenticated_tenant(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    a_docs = {upload_document(client, a, f"a{i}.txt", SHARED_POLICY)["id"] for i in range(2)}
    b_docs = {upload_document(client, b, "b.txt", SHARED_POLICY)["id"]}

    assert {d["id"] for d in client.get("/api/v1/documents", headers=a.auth).json()} == a_docs
    assert {d["id"] for d in client.get("/api/v1/documents", headers=b.auth).json()} == b_docs


def test_deleting_another_tenants_document_is_404_and_changes_nothing(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    upload_document(client, a, "refunds.txt", SHARED_POLICY)
    b_doc = upload_document(client, b, "refunds.txt", SHARED_POLICY)
    before = chunk_ids(b.business_id)

    r = client.delete(f"/api/v1/documents/{b_doc['id']}", headers=a.auth)

    assert r.status_code == 404
    assert chunk_ids(b.business_id) == before
    assert db_rows("SELECT 1 FROM documents WHERE id = :d", d=b_doc["id"])


def test_deleting_own_document_removes_its_chunks_and_only_its_chunks(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    a_doc = upload_document(client, a, "refunds.txt", SHARED_POLICY)
    upload_document(client, b, "refunds.txt", SHARED_POLICY)
    b_before = chunk_ids(b.business_id)

    assert client.delete(f"/api/v1/documents/{a_doc['id']}", headers=a.auth).status_code == 200

    assert not db_rows("SELECT 1 FROM documents WHERE id = :d", d=a_doc["id"])
    assert not db_rows("SELECT 1 FROM knowledge_chunks WHERE document_id = :d", d=a_doc["id"])
    assert chunk_ids(b.business_id) == b_before


def test_upload_ownership_comes_from_the_token_never_from_the_request_body(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")

    upload_document(client, a, "x.txt", SHARED_POLICY, business_id=str(b.business_id), tenant_id=str(b.business_id))

    assert {r["business_id"] for r in db_rows("SELECT business_id::text AS business_id FROM documents")} == {str(a.business_id)}
    assert {r["business_id"] for r in db_rows("SELECT business_id::text AS business_id FROM knowledge_chunks")} == {str(a.business_id)}
    assert not db_rows("SELECT 1 FROM documents WHERE business_id = :b", b=str(b.business_id))


# ---------------------------------------------------------------- conversations

def test_conversation_list_and_detail_are_scoped_to_the_authenticated_tenant(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    chat_once(client, a, "hello from alpha SECRETAAA")
    chat_once(client, b, "hello from bravo SECRETBBB")
    a_call = db_rows("SELECT id::text AS id FROM calls WHERE business_id = :b", b=str(a.business_id))[0]["id"]
    b_call = db_rows("SELECT id::text AS id FROM calls WHERE business_id = :b", b=str(b.business_id))[0]["id"]

    listed = {c["id"] for c in client.get("/api/v1/conversations", headers=a.auth).json()}
    assert listed == {a_call}

    foreign = client.get(f"/api/v1/conversations/{b_call}", headers=a.auth)
    assert foreign.status_code == 404
    assert "SECRETBBB" not in foreign.text

    own = client.get(f"/api/v1/conversations/{a_call}", headers=a.auth)
    assert own.status_code == 200
    messages = own.json()["messages"]
    assert {m["role"] for m in messages} == {"customer", "agent"}  # (row order within a turn is a separate, tracked issue)
    assert any("SECRETAAA" in m["content"] for m in messages)
    assert "SECRETBBB" not in own.text


def test_same_session_id_under_two_tenants_gets_separate_calls_and_separate_redis_history(client, llm):
    """The server never issues colliding session ids, but if one ever collided
    (replay, bug, future code path) the tenant boundary must still hold."""
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    sid = "shared-session-id-0123456789"

    def turn(tenant, message):
        ticket = forge_token("chat", business_id=tenant.business_id, sid=sid)
        with client.websocket_connect(f"/api/v1/voice/stream/{sid}?token={ticket}") as ws:
            ws.send_json({"text": message})
            return ws.receive_json()

    turn(a, "A-private: my card ending 4417 was charged twice")
    turn(b, "B asks about shipping times")
    turn(b, "and B follows up about returns")

    calls = db_rows("SELECT business_id::text AS business_id, caller_ref FROM calls")
    assert len(calls) == 2
    assert {c["caller_ref"] for c in calls} == {sid}
    assert {c["business_id"] for c in calls} == {str(a.business_id), str(b.business_id)}

    keys = redis_keys()
    # key layout: shrivox:session:v2:<business_id>:<session_id> -- parse from the end so a prefix change can't skew it
    assert sorted(k.split(":")[-2] for k in keys) == sorted([str(a.business_id), str(b.business_id)])

    prompts = llm.generation_prompts()
    assert len(prompts) == 3
    assert all("card ending 4417" not in p for p in prompts[1:]), "tenant A's text leaked into tenant B's prompt"
    assert "B asks about shipping" in prompts[2], "positive control: B's own history reaches B's second turn"


def test_generation_prompt_contains_only_the_callers_knowledge(client, llm):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    upload_document(client, a, "returns.txt", "Our return window is thirty days for all returns. Code ALPHA-POLICY-111 applies.")
    upload_document(client, b, "returns.txt", "Our return window is fourteen days for all returns. Code BRAVO-POLICY-222 applies.")
    question = "what is the return window for returns"

    _, reply_a = chat_once(client, a, question)
    _, reply_b = chat_once(client, b, question)

    prompt_a, prompt_b = llm.generation_prompts()
    assert "ALPHA-POLICY-111" in prompt_a and "BRAVO-POLICY-222" not in prompt_a
    assert "BRAVO-POLICY-222" in prompt_b and "ALPHA-POLICY-111" not in prompt_b
    assert reply_a["escalate"] is False and reply_b["escalate"] is False


def test_each_tenants_session_binds_to_its_own_call_row(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    a_sid, _ = chat_once(client, a, "first message")
    b_sid, _ = chat_once(client, b, "first message")
    assert a_sid != b_sid

    rows = db_rows("SELECT business_id::text AS business_id, caller_ref, (SELECT count(*) FROM messages m WHERE m.call_id = c.id) AS n FROM calls c")
    by_tenant = {r["business_id"]: r for r in rows}
    assert by_tenant[str(a.business_id)]["caller_ref"] == a_sid and by_tenant[str(a.business_id)]["n"] == 2
    assert by_tenant[str(b.business_id)]["caller_ref"] == b_sid and by_tenant[str(b.business_id)]["n"] == 2
    uuid.UUID(by_tenant[str(a.business_id)]["business_id"])  # sanity: real UUIDs
