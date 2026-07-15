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
         patch("app.agents.graph.retrieve", new=AsyncMock(return_value=[])):

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
