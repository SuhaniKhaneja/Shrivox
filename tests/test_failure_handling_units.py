"""
Unit tests for explicit failure handling (audit H5, H6, M7) -- no infrastructure.

Covers the shared upstream-call policy (classification, bounded retry, timeout,
cancellation, log hygiene), the Gemini wrappers built on it, embedding
batching, the WebSocket frame parser, atomic session-memory writes, and the
call-finalization helper.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError
from structlog.testing import capture_logs

from app.config import Settings, settings
from llm_fakes import ScriptedEmbedder, ScriptedGemini, api_error, default_vector

SECRET = "SECRET-CUSTOMER-TEXT-9931"


@pytest.fixture(autouse=True)
def fast_policy(monkeypatch):
    monkeypatch.setattr(settings, "upstream_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "upstream_max_attempts", 3)
    monkeypatch.setattr(settings, "upstream_retry_backoff_seconds", 0)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ classification

@pytest.mark.parametrize("exc,expected", [
    (api_error(400), "UpstreamRejected"), (api_error(401), "UpstreamRejected"),
    (api_error(403), "UpstreamRejected"), (api_error(404), "UpstreamRejected"),
    (api_error(408), "UpstreamUnavailable"), (api_error(429), "UpstreamUnavailable"),
    (api_error(500), "UpstreamUnavailable"), (api_error(503), "UpstreamUnavailable"),
    (httpx.ConnectError("refused"), "UpstreamUnavailable"),
    (httpx.ReadTimeout("slow"), "UpstreamUnavailable"),
    (TimeoutError(), "UpstreamUnavailable"),
])
def test_upstream_errors_are_classified(exc, expected):
    from app.core.upstream import call_with_retry

    async def fn():
        raise exc

    with pytest.raises(Exception) as caught:
        run(call_with_retry("llm", fn))
    assert type(caught.value).__name__ == expected
    assert caught.value.service == "llm"
    assert caught.value.retryable is (expected == "UpstreamUnavailable")


def test_unrelated_exceptions_pass_through_unwrapped_and_are_not_retried():
    from app.core.upstream import call_with_retry
    calls = []

    async def fn():
        calls.append(1)
        raise ValueError("programming error")

    with pytest.raises(ValueError):
        run(call_with_retry("llm", fn))
    assert len(calls) == 1


# ----------------------------------------------------------------- retrying

def test_transient_errors_are_retried_up_to_the_limit_then_raised_as_unavailable():
    from app.core.upstream import UpstreamUnavailable, call_with_retry
    calls = []

    async def fn():
        calls.append(1)
        raise api_error(503)

    with pytest.raises(UpstreamUnavailable):
        run(call_with_retry("llm", fn))
    assert len(calls) == 3


def test_client_errors_are_never_retried():
    from app.core.upstream import UpstreamRejected, call_with_retry
    calls = []

    async def fn():
        calls.append(1)
        raise api_error(400)

    with pytest.raises(UpstreamRejected):
        run(call_with_retry("llm", fn))
    assert len(calls) == 1


def test_a_transient_error_followed_by_success_is_invisible_to_the_caller():
    from app.core.upstream import call_with_retry
    outcomes = [api_error(429), "fine"]

    async def fn():
        o = outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o

    assert run(call_with_retry("llm", fn)) == "fine"


def test_a_slow_call_times_out_and_is_retried_within_a_bounded_time():
    from app.core.upstream import UpstreamUnavailable, call_with_retry
    calls = []

    async def fn():
        calls.append(1)
        await asyncio.sleep(5)

    started = time.perf_counter()
    with pytest.raises(UpstreamUnavailable):
        run(call_with_retry("llm", fn))
    assert len(calls) == 3
    assert time.perf_counter() - started < 2


def test_cancellation_propagates_immediately_and_is_not_retried():
    from app.core.upstream import call_with_retry
    calls = []

    async def fn():
        calls.append(1)
        await asyncio.sleep(5)

    async def scenario():
        task = asyncio.create_task(call_with_retry("llm", fn))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert len(calls) == 1


def test_retry_logs_carry_service_attempt_and_error_type_but_never_the_error_text():
    from app.core.upstream import UpstreamUnavailable, call_with_retry

    async def fn():
        raise api_error(503, message=SECRET)

    with capture_logs() as logs, pytest.raises(UpstreamUnavailable):
        run(call_with_retry("llm", fn))

    retries = [l for l in logs if l["event"] == "upstream_retry"]
    assert [r["attempt"] for r in retries] == [1, 2]
    assert all(r["service"] == "llm" and r["status_code"] == 503 and r["error_type"] == "ServerError" for r in retries)
    assert SECRET not in str(logs)


# --------------------------------------------------------- chat_completion

def _chat(gemini, messages=None):
    from app.agents.llm_client import chat_completion
    with patch("app.agents.llm_client._client", gemini):
        return run(chat_completion(messages or [{"role": "user", "content": "hi"}]))


@pytest.mark.parametrize("empty", [None, "", "   \n"])
def test_empty_model_output_raises_a_typed_error_without_retrying(empty):
    from app.core.upstream import UpstreamEmptyResponse
    g = ScriptedGemini([empty])
    with pytest.raises(UpstreamEmptyResponse):
        _chat(g)
    assert g.attempts == 1


def test_chat_completion_bad_request_is_not_retried():
    from app.core.upstream import UpstreamRejected
    g = ScriptedGemini([api_error(400)])
    with pytest.raises(UpstreamRejected):
        _chat(g)
    assert g.attempts == 1


def test_chat_completion_recovers_from_a_transient_error():
    g = ScriptedGemini([api_error(503), "second time lucky"])
    assert _chat(g)["text"] == "second time lucky"
    assert g.attempts == 2


def test_intent_classifier_empty_output_degrades_to_the_default_intent_instead_of_failing_the_turn():
    from app.agents.graph import shrivox_agent
    g = ScriptedGemini([None, "Here is your answer."])
    state = {"session_id": "s", "business_id": "00000000-0000-0000-0000-000000000001", "user_message": "hello",
             "conversation_history": [], "intent": None, "retrieved_docs": [], "retrieved_doc_ids": [],
             "response_text": None, "safety_flag": None, "escalate": False, "latency_ms": 0, "token_count": 0}

    async def go():
        with patch("app.agents.llm_client._client", g), patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):
            return await shrivox_agent.ainvoke(state)

    out = run(go())
    assert out["intent"] == "general_query"
    assert out["response_text"] == "Here is your answer."


# --------------------------------------------------------------- embeddings

def test_embed_texts_sends_one_call_per_batch_and_preserves_order(monkeypatch):
    from app.retrieval import retriever
    monkeypatch.setattr(retriever, "EMBED_BATCH_SIZE", 32)
    e = ScriptedEmbedder()
    texts = [f"chunk number {i}" for i in range(70)]
    with patch("app.retrieval.retriever._genai_client", e):
        vectors = run(retriever.embed_texts(texts))
    assert [len(c) for c in e.calls] == [32, 32, 6]
    assert vectors == [default_vector(t) for t in texts]


def test_embed_texts_falls_back_to_single_calls_when_the_api_returns_the_wrong_count(monkeypatch):
    from app.retrieval import retriever
    monkeypatch.setattr(retriever, "EMBED_BATCH_SIZE", 3)
    e = ScriptedEmbedder(steps=["short"])
    texts = ["alpha one", "beta two", "gamma three"]
    with patch("app.retrieval.retriever._genai_client", e), capture_logs() as logs:
        vectors = run(retriever.embed_texts(texts))
    assert vectors == [default_vector(t) for t in texts]          # still correct and aligned
    assert [len(c) for c in e.calls] == [3, 1, 1, 1]
    assert [l for l in logs if l["event"] == "embed_batch_count_mismatch"]


def test_embed_text_persistent_failure_is_a_typed_embedding_error():
    from app.core.upstream import UpstreamUnavailable
    from app.retrieval import retriever
    e = ScriptedEmbedder(steps=[api_error(503)] * 3)
    with patch("app.retrieval.retriever._genai_client", e), pytest.raises(UpstreamUnavailable) as caught:
        run(retriever.embed_text("hello", task_type="RETRIEVAL_QUERY"))
    assert caught.value.service == "embedding"


# ------------------------------------------------------- WebSocket frame parser

@pytest.mark.parametrize("raw,code", [
    ("not json", "invalid_frame"),
    ("[1, 2]", "invalid_frame"),
    ('"just a string"', "invalid_frame"),
    ("{}", "invalid_message"),
    ('{"text": 123}', "invalid_message"),
    ('{"text": "   "}', "invalid_message"),
    ('{"text": null}', "invalid_message"),
    ('{"text": "' + "x" * 51 + '"}', "message_too_long"),
    ("x" * 5000, "invalid_frame"),
])
def test_frame_parser_rejects_bad_frames_with_specific_codes(raw, code, monkeypatch):
    from app.api.conversations import FrameError, parse_client_frame
    monkeypatch.setattr(settings, "max_message_chars", 50)
    with pytest.raises(FrameError) as caught:
        parse_client_frame(raw)
    assert caught.value.code == code


def test_frame_parser_accepts_and_strips_a_valid_message(monkeypatch):
    from app.api.conversations import parse_client_frame
    monkeypatch.setattr(settings, "max_message_chars", 50)
    assert parse_client_frame('{"text": "  mera order kahan hai?  ", "business_id": "ignored"}') == "mera order kahan hai?"
    assert parse_client_frame('{"text": "' + "x" * 50 + '"}') == "x" * 50


# ----------------------------------------------------------- session memory

def test_append_turns_writes_both_messages_in_a_single_redis_transaction():
    import uuid
    from app.memory import session_memory
    from auth_helpers import FakeRedis
    r, biz = FakeRedis(), uuid.uuid4()
    with patch("app.memory.session_memory.get_redis", return_value=r):
        history = run(session_memory.append_turns(biz, "s1", [("customer", "q"), ("agent", "a")]))
    assert r.executed_pipelines == 1
    assert [t["role"] for t in history] == ["customer", "agent"]


def test_append_turns_still_bounds_the_window(monkeypatch):
    import uuid
    from app.memory import session_memory
    from auth_helpers import FakeRedis
    monkeypatch.setattr(settings, "max_conversation_turns", 4)
    r, biz = FakeRedis(), uuid.uuid4()
    with patch("app.memory.session_memory.get_redis", return_value=r):
        for i in range(5):
            run(session_memory.append_turns(biz, "s1", [("customer", f"q{i}"), ("agent", f"a{i}")]))
    stored = [json.loads(x) for x in next(iter(r.store.values()))]
    assert [t["content"] for t in stored] == ["q3", "a3", "q4", "a4"]


# --------------------------------------------------------- call finalization

def test_finalize_call_never_raises_even_if_the_database_is_gone():
    import uuid
    from app.api.conversations import _finalize_call
    db = MagicMock()
    db.execute = AsyncMock(side_effect=OperationalError("stmt", {}, Exception("connection lost")))
    db.rollback = AsyncMock()
    with capture_logs() as logs:
        run(_finalize_call(db, uuid.uuid4(), "disconnected"))
    assert [l["error_type"] for l in logs if l["event"] == "call_finalize_failed"] == ["OperationalError"]


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("field,bad", [
    ("upstream_max_attempts", 0), ("upstream_max_attempts", 9), ("upstream_timeout_seconds", 0),
    ("turn_timeout_seconds", 1), ("max_message_chars", 0), ("max_document_chars", 10), ("redis_timeout_seconds", 0),
])
def test_failure_policy_settings_are_bounded(field, bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, google_api_key="k", jwt_secret="s", **{field: bad})
