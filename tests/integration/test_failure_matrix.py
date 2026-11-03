"""
Failure-injection matrix against REAL PostgreSQL + Redis (audit H5, H6, M7).

Gemini is scripted at the SDK layer (tests/llm_fakes.py), so the production
classification, retry, timeout and empty-response code is what reacts. Each
test asserts the four things a failure must get right:

  1. the CLIENT sees a typed, internals-free error (and the connection survives)
  2. POSTGRES holds exactly what was documented (no partial turn / document)
  3. REDIS agrees with Postgres
  4. the LOG has a structured event and never customer text or exception text

The tests are written as the specification; the behaviour table in the README
("Failure handling") is derived from them.
"""

import asyncio

import pytest
import redis.exceptions
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from starlette.websockets import WebSocketDisconnect
from structlog.testing import capture_logs
from unittest.mock import AsyncMock

from app.config import settings
from app.db.session import get_db
from app.main import app
from app.retrieval.chunking import chunk_text
from integration_helpers import PerCallRedis, db_rows, fake_embedding, redis_keys, register_tenant
from llm_fakes import ScriptedEmbedder, ScriptedGemini, api_error

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(settings, "upstream_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "upstream_max_attempts", 2)
    monkeypatch.setattr(settings, "upstream_retry_backoff_seconds", 0)
    monkeypatch.setattr(settings, "turn_timeout_seconds", 1.5)
    monkeypatch.setattr(settings, "max_message_chars", 50)


@pytest.fixture
def gemini(monkeypatch, policy):
    g = ScriptedGemini()
    monkeypatch.setattr("app.agents.llm_client._client", g)
    return g


@pytest.fixture
def embedder(monkeypatch, policy):
    e = ScriptedEmbedder(vector_fn=fake_embedding)
    monkeypatch.setattr("app.retrieval.retriever._genai_client", e)
    return e


@pytest.fixture
def client(gemini, embedder):
    return TestClient(app)


@pytest.fixture
def tenant(client):
    return register_tenant(client, "Alpha")


async def slow():
    await asyncio.sleep(5)


def chat_url(client, tenant):
    issued = client.post("/api/v1/auth/chat-ticket", headers=tenant.auth).json()
    return f"{issued['ws_path']}?token={issued['ticket']}"


def turn(ws, text="where is my order"):
    ws.send_json({"text": text})
    return ws.receive_json()


def messages(tenant) -> int:
    return db_rows("SELECT count(*) AS n FROM messages m JOIN calls c ON c.id = m.call_id WHERE c.business_id = :b",
                   b=str(tenant.business_id))[0]["n"]


def assert_error(frame, code, retryable):
    assert frame["type"] == "error"
    assert frame["code"] == code
    assert frame["retryable"] is retryable
    assert set(frame) == {"type", "code", "message", "retryable"}


# ============================================================ chat: LLM failures

def test_llm_timeout_gives_a_retryable_error_leaves_no_partial_state_and_the_session_recovers(client, tenant, gemini):
    gemini.steps = ["general_query", slow, slow]            # intent ok, then generation times out on both attempts
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "llm_unavailable", True)
        assert gemini.attempts == 3                          # 1 intent + 2 generation attempts (bounded)
        assert messages(tenant) == 0 and redis_keys() == []  # nothing half-written anywhere

        ok = turn(ws)                                        # same connection, same session
        assert ok["type"] == "response" and "degraded" not in ok
    assert messages(tenant) == 2 and len(redis_keys()) == 1


def test_rate_limit_followed_by_success_is_invisible_to_the_customer(client, tenant, gemini):
    gemini.steps = ["general_query", api_error(429), "Recovered answer"]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        frame = turn(ws)
    assert frame["response"] == "Recovered answer"
    assert gemini.attempts == 3


@pytest.mark.parametrize("failure", [api_error(429), api_error(503)])
def test_persistent_transient_errors_surface_as_llm_unavailable(client, tenant, gemini, failure):
    gemini.steps = ["general_query", failure, failure]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "llm_unavailable", True)
    assert messages(tenant) == 0


def test_bad_request_is_not_retried_and_is_not_retryable(client, tenant, gemini):
    gemini.steps = ["general_query", api_error(400)]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "llm_rejected", False)
    assert gemini.attempts == 2                              # intent + exactly ONE generation attempt
    assert messages(tenant) == 0


def test_empty_generation_is_reported_not_sent_as_an_empty_answer(client, tenant, gemini):
    gemini.steps = ["general_query", None]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "llm_empty_response", True)
    assert messages(tenant) == 0 and redis_keys() == []


def test_empty_intent_output_degrades_to_the_default_intent_and_the_turn_succeeds(client, tenant, gemini):
    gemini.steps = [None, "Your order ships tomorrow."]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        frame = turn(ws)
    assert frame["response"] == "Your order ships tomorrow." and frame["intent"] == "general_query"
    assert {r["intent"] for r in db_rows("SELECT intent FROM messages WHERE role = 'customer'")} == {"general_query"}


# =========================================================== chat: retrieval failures

def test_query_embedding_failure_blocks_the_answer_instead_of_fabricating_one(client, tenant, gemini, embedder):
    embedder.steps = [api_error(503), api_error(503)]
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "embedding_unavailable", True)
    assert gemini.attempts == 1                              # the intent call only: generation never ran
    assert messages(tenant) == 0 and redis_keys() == []


def test_database_failure_during_retrieval_is_reported_and_generation_never_runs(client, tenant, gemini, monkeypatch):
    monkeypatch.setattr("app.agents.graph.retrieve", AsyncMock(side_effect=OperationalError("SELECT secret_column", {}, Exception("connection lost"))))
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        frame = turn(ws)
    assert_error(frame, "database_unavailable", True)
    assert "secret_column" not in str(frame)
    assert gemini.attempts == 1 and messages(tenant) == 0


# ====================================================== chat: persistence ordering

def test_database_failure_while_saving_withholds_the_reply_and_keeps_redis_consistent(client, tenant, monkeypatch):
    """The failure is REAL, not a raised exception: the session's own Postgres
    connection is killed mid-turn, exactly what a database restart or failover
    looks like. The session is then in a failed state (PendingRollbackError on
    its next statement) until the handler rolls it back, so this test also
    proves the next message on the same connection works."""
    from sqlalchemy import text
    from app.api import conversations
    real, calls = conversations._persist_turn, {"n": 0}

    async def persist_with_killed_connection(db, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            await db.execute(text("SELECT pg_terminate_backend(pg_backend_pid())"))
        return await real(db, *args, **kwargs)

    monkeypatch.setattr("app.api.conversations._persist_turn", persist_with_killed_connection)
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        frame = turn(ws, "first attempt")
        assert_error(frame, "database_unavailable", True)
        assert "pg_terminate" not in str(frame) and "terminating" not in str(frame)
        assert redis_keys() == [] and messages(tenant) == 0   # a reply that was never saved was never remembered either

        ok = turn(ws, "second attempt")                       # same session: the failed transaction was rolled back
        assert ok["type"] == "response"
    assert messages(tenant) == 2 and len(redis_keys()) == 1


def test_redis_outage_degrades_the_turn_but_postgres_stays_complete(client, tenant, monkeypatch):
    class FlakyRedis(PerCallRedis):
        down = True

        async def _run(self, op):
            if self.down:
                raise redis.exceptions.ConnectionError("redis down")
            return await super()._run(op)

    flaky = FlakyRedis()
    monkeypatch.setattr("app.memory.session_memory.get_redis", lambda: flaky)

    with capture_logs() as logs, client.websocket_connect(chat_url(client, tenant)) as ws:
        first = turn(ws, "while redis is down")
        assert first["type"] == "response" and first["degraded"] == ["session_memory"]
        assert messages(tenant) == 2                          # the durable record is complete

        flaky.down = False
        second = turn(ws, "after redis recovers")
        assert second["type"] == "response" and "degraded" not in second
    assert messages(tenant) == 4 and len(redis_keys()) == 1
    assert {l["op"] for l in logs if l["event"] == "redis_unavailable"} == {"get_history", "append_turns"}


def test_database_unreachable_when_the_session_opens_sends_an_error_then_closes_1011(client, tenant, monkeypatch):
    monkeypatch.setattr("app.api.conversations._get_or_create_call", AsyncMock(side_effect=ConnectionRefusedError()))
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(ws.receive_json(), "database_unavailable", True)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1011


# ============================================================ chat: protocol errors

@pytest.mark.parametrize("send,code", [
    (lambda ws: ws.send_text("not json"), "invalid_frame"),
    (lambda ws: ws.send_text("[1, 2]"), "invalid_frame"),
    (lambda ws: ws.send_bytes(b"\x00\x01\x02"), "invalid_frame"),
    (lambda ws: ws.send_json({}), "invalid_message"),
    (lambda ws: ws.send_json({"text": 123}), "invalid_message"),
    (lambda ws: ws.send_json({"text": "   "}), "invalid_message"),
    (lambda ws: ws.send_json({"text": "x" * 60}), "message_too_long"),
])
def test_bad_frames_get_typed_errors_and_never_kill_the_connection(client, tenant, send, code):
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        send(ws)
        assert_error(ws.receive_json(), code, False)
        assert messages(tenant) == 0 and redis_keys() == []   # rejected input costs nothing

        assert turn(ws)["type"] == "response"                 # the same connection still works
    assert messages(tenant) == 2


def test_unexpected_exceptions_are_masked_from_the_client_and_the_logs(client, tenant, monkeypatch, capsys, caplog):
    from app.api import conversations
    real, calls = conversations.shrivox_agent.ainvoke, {"n": 0}

    async def exploding_agent(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("INTERNAL-SECRET-DETAIL for customer jane@example.com")
        return await real(state)

    monkeypatch.setattr("app.api.conversations.shrivox_agent.ainvoke", exploding_agent)
    with capture_logs() as logs, client.websocket_connect(chat_url(client, tenant)) as ws:
        frame = turn(ws, "my email is jane@example.com")
        assert_error(frame, "internal_error", False)
        assert turn(ws)["type"] == "response"                 # and the session carries on

    blob = str(frame) + str(logs) + caplog.text + "".join(capsys.readouterr())
    assert "INTERNAL-SECRET-DETAIL" not in blob and "jane@example.com" not in blob
    assert [l["error_type"] for l in logs if l["event"] == "turn_failed"] == ["RuntimeError"]


def test_a_turn_that_runs_too_long_is_cut_off_at_the_deadline(client, tenant, monkeypatch):
    from app.api import conversations
    real, calls = conversations.shrivox_agent.ainvoke, {"n": 0}

    async def hanging_agent(state):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(10)
        return await real(state)

    monkeypatch.setattr("app.api.conversations.shrivox_agent.ainvoke", hanging_agent)
    monkeypatch.setattr(settings, "turn_timeout_seconds", 0.3)
    with client.websocket_connect(chat_url(client, tenant)) as ws:
        assert_error(turn(ws), "turn_timeout", True)
        assert turn(ws)["type"] == "response"


# Call lifecycle (completed / escalated / reconnect) is tested against a REAL uvicorn
# server in test_live_server_lifecycle.py: Starlette's TestClient cancels the app the
# moment the client closes, which would cut off the very teardown being tested.


# =============================================================== ingestion

def long_text(words: int = 1100) -> str:
    return " ".join(f"word{i}" for i in range(words))


def doc_and_chunk_counts() -> tuple[int, int]:
    return (db_rows("SELECT count(*) AS n FROM documents")[0]["n"],
            db_rows("SELECT count(*) AS n FROM knowledge_chunks")[0]["n"])


def upload(client, tenant, text, filename="kb.txt"):
    return client.post("/api/v1/documents/upload", headers=tenant.auth, json={"filename": filename, "text": text})


def test_embedding_failure_midway_leaves_no_document_and_no_chunks(client, tenant, embedder, monkeypatch):
    monkeypatch.setattr("app.retrieval.retriever.EMBED_BATCH_SIZE", 2)
    text = long_text()
    assert len(chunk_text(text)) >= 3
    embedder.steps = ["ok", api_error(503), api_error(503)]   # first batch fine, second batch fails both attempts

    r = upload(client, tenant, text)

    assert r.status_code == 503 and r.json()["code"] == "embedding_unavailable"
    assert doc_and_chunk_counts() == (0, 0)                   # nothing stuck in 'processing', nothing orphaned


def test_embedding_rejection_is_a_502_and_is_not_retried(client, tenant, embedder):
    embedder.steps = [api_error(400)]
    r = upload(client, tenant, long_text(100))
    assert r.status_code == 502 and r.json()["code"] == "embedding_rejected"
    assert len(embedder.calls) == 1 and doc_and_chunk_counts() == (0, 0)


def test_a_transient_embedding_error_is_retried_and_the_upload_succeeds(client, tenant, embedder):
    embedder.steps = [api_error(503), "ok"]
    r = upload(client, tenant, long_text(100))
    assert r.status_code == 201
    assert doc_and_chunk_counts() == (1, r.json()["chunks_created"])


def test_ingestion_is_atomic_when_the_database_rejects_a_chunk(client, tenant, monkeypatch):
    """Earlier chunks insert fine; the LAST has a wrong-sized vector, so Postgres
    rejects it. The document and every earlier chunk must roll back with it."""
    text = long_text()
    n = len(chunk_text(text))
    assert n >= 3
    bad_embedder = ScriptedEmbedder(vector_fn=lambda t: [0.1, 0.2, 0.3] if "word1099" in t else fake_embedding(t))
    monkeypatch.setattr("app.retrieval.retriever._genai_client", bad_embedder)

    r = upload(client, tenant, text)

    assert r.status_code == 500 and r.json()["code"] == "internal_error"
    assert "dimensions" not in r.text                          # no Postgres internals leak to the client
    assert doc_and_chunk_counts() == (0, 0)


def test_embedding_uses_one_call_per_batch_not_one_per_chunk(client, tenant, embedder, monkeypatch):
    monkeypatch.setattr("app.retrieval.retriever.EMBED_BATCH_SIZE", 2)
    text = long_text()
    n = len(chunk_text(text))
    r = upload(client, tenant, text)
    assert r.status_code == 201 and r.json()["chunks_created"] == n
    assert [len(c) for c in embedder.calls] == [2] * (n // 2) + ([1] if n % 2 else [])


def test_a_wrong_sized_batch_response_falls_back_to_single_calls_and_still_succeeds(client, tenant, embedder, monkeypatch):
    monkeypatch.setattr("app.retrieval.retriever.EMBED_BATCH_SIZE", 3)
    embedder.steps = ["short"]
    text = long_text()
    n = len(chunk_text(text))
    r = upload(client, tenant, text)
    assert r.status_code == 201 and doc_and_chunk_counts() == (1, n)


@pytest.mark.parametrize("payload", [
    {"filename": "kb.txt", "text": "   \n\t "},
    {"filename": "kb.txt", "text": "x" * 3000},
    {"filename": "", "text": "hello world"},
    {"filename": "f" * 256, "text": "hello world"},
])
def test_invalid_uploads_are_rejected_before_any_embedding_or_write(client, tenant, embedder, monkeypatch, payload):
    monkeypatch.setattr(settings, "max_document_chars", 2000)
    r = client.post("/api/v1/documents/upload", headers=tenant.auth, json=payload)
    assert r.status_code == 422
    assert embedder.calls == [] and doc_and_chunk_counts() == (0, 0)


# ======================================================== REST: infrastructure down

@pytest.mark.parametrize("failure", [
    OperationalError("SELECT secret_table", {}, Exception("connection lost")),
    ConnectionRefusedError(),
])
def test_rest_endpoints_report_a_database_outage_as_503_without_internals(client, tenant, failure):
    class DeadSession:
        async def execute(self, *a, **kw):
            raise failure

    async def dead_db():
        yield DeadSession()

    app.dependency_overrides[get_db] = dead_db
    try:
        r = client.get("/api/v1/documents", headers=tenant.auth)
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert r.status_code == 503 and r.json()["code"] == "database_unavailable"
    assert "secret_table" not in r.text
