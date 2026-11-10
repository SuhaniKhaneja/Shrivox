"""
Tests the graph's branching logic without hitting real Gemini/Postgres.
We mock the external calls (chat_completion, retrieve) so
these tests are fast, free, and deterministic -- they check that the
*graph wiring* is correct (escalation short-circuits retrieval+generation,
safety retry fires when expected), not that the LLM produces good text.
That second thing is exactly what run_eval.py is for.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, patch

from app.agents.graph import shrivox_agent

TEST_BUSINESS_ID = uuid.uuid4()


def base_state(user_message: str) -> dict:
    return {
        "session_id": "test-session",
        "business_id": TEST_BUSINESS_ID,
        "user_message": user_message,
        "conversation_history": [],
        "intent": None,
        "retrieved_docs": [],
        "retrieved_doc_ids": [],
        "response_text": None,
        "safety_flag": None,
        "escalate": False,
        "latency_ms": 0,
        "token_count": 0,
    }


@pytest.mark.asyncio
async def test_escalation_trigger_skips_llm_calls():
    """A legal-threat message should never reach chat_completion or retrieve."""
    with patch("app.agents.graph.chat_completion", new=AsyncMock()) as mock_chat, \
         patch("app.agents.graph.retrieve", new=AsyncMock()) as mock_retrieve:

        result = await shrivox_agent.ainvoke(base_state("I will sue you and get a lawyer"))

        assert result["escalate"] is True
        assert "human agent" in result["response_text"].lower()
        mock_chat.assert_not_called()
        mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_normal_flow_calls_intent_retrieval_and_generation():
    mock_intent_resp = {"text": "refund_request", "latency_ms": 10, "prompt_tokens": 5, "completion_tokens": 2}
    mock_gen_resp = {"text": "Your refund will process in 5-7 days.", "latency_ms": 200, "prompt_tokens": 50, "completion_tokens": 15}

    with patch("app.agents.graph.chat_completion", new=AsyncMock(side_effect=[mock_intent_resp, mock_gen_resp])), \
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[REFUND_POLICY])):

        result = await shrivox_agent.ainvoke(base_state("How long for my refund?"))

        assert result["escalate"] is False
        assert result["intent"] == "refund_request"
        assert "refund" in result["response_text"].lower()


@pytest.mark.asyncio
async def test_ungrounded_promise_triggers_retry():
    mock_intent_resp = {"text": "refund_request", "latency_ms": 10, "prompt_tokens": 5, "completion_tokens": 2}
    ungrounded_resp = {"text": "I guarantee you double refund!", "latency_ms": 200, "prompt_tokens": 50, "completion_tokens": 15}
    retry_resp = {"text": "I can check the refund policy for you.", "latency_ms": 180, "prompt_tokens": 55, "completion_tokens": 12}

    with patch("app.agents.graph.chat_completion", new=AsyncMock(side_effect=[mock_intent_resp, ungrounded_resp, retry_resp])), \
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):  # empty -> not grounded

        result = await shrivox_agent.ainvoke(base_state("Can you guarantee double refund?"))

        assert result["safety_flag"] == "ungrounded_promise"
        assert result["response_text"] == "I can check the refund policy for you."


# ============================================================================
# Graph topology and control flow (audit: the graph used to be a straight line)
#
# The README and the module docstring described conditional branches, but the
# compiled graph had ZERO conditional edges: escalation was an `if` buried inside
# each node, a classifier verdict of `escalation_request` was ignored, and intent
# classification and retrieval -- independent of each other -- ran back to back.
# ============================================================================

import asyncio


def _doc(text, doc_id="11111111-1111-1111-1111-111111111111"):
    """A retrieved KB passage, as retrieve() returns it."""
    return type("RetrievedDoc", (), {"text": text, "id": doc_id})()


REFUND_POLICY = _doc("Refunds are processed within 5-7 business days after we receive the item.")


def _resp(text, prompt=5, completion=2):
    return {"text": text, "latency_ms": 10, "prompt_tokens": prompt, "completion_tokens": completion}


def test_the_graph_has_real_conditional_edges_and_a_parallel_fan_out():
    edges = shrivox_agent.get_graph().edges
    conditional = {(e.source, e.target) for e in edges if e.conditional}

    # escalation branches off BEFORE any I/O, and again after the classifier
    assert ("input_guard", "escalate_response") in conditional
    assert ("triage", "escalate_response") in conditional
    assert ("triage", "generate_response") in conditional
    # the output check can loop back to regenerate, bail out to a fallback, or finish
    assert {("safety_check", "regenerate_response"), ("safety_check", "safe_fallback")} <= conditional
    # intent classification and retrieval are siblings: both start from the guard
    fan_out = {t for s, t in conditional if s == "input_guard"}
    assert {"classify_intent", "retrieve_context"} <= fan_out


@pytest.mark.asyncio
async def test_intent_classification_and_retrieval_run_concurrently():
    """Each branch waits (bounded) for the OTHER to have started. In a sequential
    graph the first branch would wait for one that never starts, time out, and fail
    this test -- so it passes only if the two genuinely overlap. No sleeps, no
    wall-clock thresholds, so it cannot flake."""
    classifier_started, retrieval_started = asyncio.Event(), asyncio.Event()

    async def chat(messages, **kwargs):
        if "Classify the customer's message" in messages[0]["content"]:
            classifier_started.set()
            await asyncio.wait_for(retrieval_started.wait(), timeout=2)
            return _resp("refund_request")
        return _resp("Refunds take 5-7 days.")

    async def retrieve(db, business_id, query, **kwargs):
        retrieval_started.set()
        await asyncio.wait_for(classifier_started.wait(), timeout=2)
        return [REFUND_POLICY]

    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=retrieve):
        result = await shrivox_agent.ainvoke(base_state("How long for my refund?"))

    assert result["intent"] == "refund_request" and result["escalate"] is False


@pytest.mark.asyncio
async def test_a_classifier_verdict_of_escalation_request_escalates_and_skips_generation():
    """Previously the classifier's `escalation_request` was recorded as the intent but
    escalate stayed False, so the customer who asked for a human got a normal answer
    (and a full generation call was spent on it)."""
    chat = AsyncMock(side_effect=[_resp("escalation_request"), AssertionError("generation must not run")])
    retrieved = [type("D", (), {"text": "t", "id": "11111111-1111-1111-1111-111111111111"})()]

    with patch("app.agents.graph.chat_completion", new=chat), \
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=retrieved)):
        # Deliberately a phrasing no keyword rule matches, so this exercises the CLASSIFIER path.
        result = await shrivox_agent.ainvoke(base_state("this bot is useless, is there anyone else who can help me"))

    assert chat.await_count == 1                       # the classifier only
    assert result["escalate"] is True
    assert result["intent"] == "escalation_request"
    assert result["safety_flag"] == "escalation_intent"
    assert "human agent" in result["response_text"].lower()
    assert result["retrieved_doc_ids"] == []           # nothing was used, so nothing is recorded as used


@pytest.mark.asyncio
async def test_a_second_unsafe_answer_is_never_shipped_it_falls_back_to_a_human():
    """Before: one retry, then whatever came back was sent -- flagged but unchecked.
    Now the retry is re-checked, and if it is still unsafe the customer gets a safe
    hand-off instead of the text."""
    chat = AsyncMock(side_effect=[
        _resp("refund_request"),
        _resp("I guarantee you double refund!"),
        _resp("Honestly I promise it will be fully refunded."),   # the retry is ALSO an ungrounded promise
    ])
    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):
        result = await shrivox_agent.ainvoke(base_state("Can you guarantee double refund?"))

    assert chat.await_count == 3                       # intent + generation + exactly ONE retry; the loop is bounded
    assert result["escalate"] is True
    assert result["safety_flag"] == "ungrounded_promise"
    assert "guarantee" not in result["response_text"].lower() and "promise" not in result["response_text"].lower()
    assert "human agent" in result["response_text"].lower()


@pytest.mark.asyncio
async def test_a_retry_that_fixes_the_answer_keeps_the_flag_but_does_not_escalate():
    chat = AsyncMock(side_effect=[_resp("refund_request"), _resp("I guarantee a refund!"), _resp("Let me check the policy for you.")])
    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):
        result = await shrivox_agent.ainvoke(base_state("Can you guarantee a refund?"))

    assert result["response_text"] == "Let me check the policy for you."
    assert result["safety_flag"] == "ungrounded_promise"   # analytics still see that the guardrail intervened
    assert result["escalate"] is False


@pytest.mark.asyncio
async def test_token_count_covers_every_llm_call_in_the_turn():
    """The classifier call and any retry cost tokens too. Counting only the first
    generation under-reported the real spend of every turn."""
    chat = AsyncMock(side_effect=[
        _resp("refund_request", 10, 1),                 # classifier   = 11
        _resp("I guarantee a refund!", 100, 20),        # generation   = 120
        _resp("Let me check that.", 110, 15),           # retry        = 125
    ])
    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):
        result = await shrivox_agent.ainvoke(base_state("Can you guarantee a refund?"))
    assert result["token_count"] == 11 + 120 + 125


@pytest.mark.asyncio
async def test_retrieval_is_given_the_previous_customer_message_for_a_short_follow_up():
    history = [{"role": "customer", "content": "mera refund kab milega"}, {"role": "agent", "content": "5-7 din"}]
    retrieve = AsyncMock(return_value=[])
    with patch("app.agents.graph.chat_completion", new=AsyncMock(side_effect=[_resp("general_query"), _resp("ok")])), \
         patch("app.agents.graph.retrieve", new=retrieve):
        await shrivox_agent.ainvoke({**base_state("aur kitne din lagenge"), "conversation_history": history})

    assert retrieve.await_args.args[2] == "aur kitne din lagenge"                       # the message itself is still searched
    assert retrieve.await_args.kwargs["context_query"] == "mera refund kab milega aur kitne din lagenge"


@pytest.mark.asyncio
async def test_retrieval_gets_no_context_on_the_first_turn():
    retrieve = AsyncMock(return_value=[])
    with patch("app.agents.graph.chat_completion", new=AsyncMock(side_effect=[_resp("general_query"), _resp("ok")])), \
         patch("app.agents.graph.retrieve", new=retrieve):
        await shrivox_agent.ainvoke(base_state("How long for my refund?"))
    assert retrieve.await_args.kwargs["context_query"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "mujhe kisi insaan se baat karni hai",          # romanised Hinglish
    "मुझे किसी इंसान से बात करनी है",                  # Devanagari
    "main consumer court jaunga",
])
async def test_hinglish_escalation_is_caught_by_the_rules_at_zero_llm_cost(message):
    """The input rules used to be English-only, so every one of these cost an LLM call
    and a retrieval and then got a normal answer."""
    chat, retrieve = AsyncMock(), AsyncMock()
    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=retrieve):
        result = await shrivox_agent.ainvoke(base_state(message))
    assert result["escalate"] is True and result["safety_flag"] == "escalation_trigger"
    chat.assert_not_awaited()
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_harm_gets_a_caring_reply_not_the_routine_handoff_line():
    chat, retrieve = AsyncMock(), AsyncMock()
    with patch("app.agents.graph.chat_completion", new=chat), patch("app.agents.graph.retrieve", new=retrieve):
        result = await shrivox_agent.ainvoke(base_state("I don't want to live anymore"))
    assert result["escalate"] is True and result["safety_flag"] == "self_harm"
    assert "sorry" in result["response_text"].lower() and "emergency" in result["response_text"].lower()
    assert result["response_text"] != "I understand -- let me connect you with a human agent who can help with this right away."
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_invented_amount_is_blocked_retried_with_a_targeted_instruction_and_then_accepted():
    """The costliest support hallucination: a number the KB never gave. Before, only the
    words 'guarantee'/'promise' were checked, so this shipped."""
    chat = AsyncMock(side_effect=[
        _resp("refund_request"),
        _resp("You will get 5000 rupees back within 2 days."),                       # invented
        _resp("Refunds are processed within 5-7 business days after we get the item."),  # grounded
    ])
    with patch("app.agents.graph.chat_completion", new=chat), \
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[REFUND_POLICY])):
        result = await shrivox_agent.ainvoke(base_state("kitna refund milega aur kab"))

    assert result["response_text"].startswith("Refunds are processed within 5-7")
    assert result["safety_flag"] == "ungrounded_specifics" and result["escalate"] is False
    retry_messages = chat.await_args_list[2].args[0]
    assert any("Do not state any amount" in m["content"] for m in retry_messages if m["role"] == "system")


@pytest.mark.asyncio
async def test_an_amount_the_customer_quoted_is_not_treated_as_an_invention():
    chat = AsyncMock(side_effect=[_resp("refund_request"), _resp("Aapke Rs 1,499 ka refund process ho jayega.")])
    with patch("app.agents.graph.chat_completion", new=chat), \
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[_doc("Refund amounts equal the price paid.")])):
        result = await shrivox_agent.ainvoke(base_state("maine 1499 diye the, refund?"))
    assert result["safety_flag"] is None and chat.await_count == 2    # no retry needed
