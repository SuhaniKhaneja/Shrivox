"""
Agent-correctness regression tests (audit H1, H3, intent validation, usage guards).

These drive the real graph and the real llm_client conversion code; only the
Gemini SDK object and retrieval are replaced, and the fake Gemini RECORDS the
exact request the SDK would have received -- so the assertions are about what
Gemini actually gets, not about an intermediate helper.
"""

import random
import string
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from structlog.testing import capture_logs

from app.agents.graph import shrivox_agent
from app.agents.llm_client import _to_gemini_contents, chat_completion
from app.prompts.templates import INTENT_DETECTION_PROMPT, SYSTEM_PROMPT, build_response_prompt


class FakeGemini:
    """Stands in for the genai client held in app.agents.llm_client._client."""

    def __init__(self, texts, usage=(10, 5), usage_metadata="default"):
        self._texts = list(texts)
        self._usage = usage
        self._usage_metadata = usage_metadata
        self.requests: list[dict] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model, contents, config):
        self.requests.append({
            "system_instruction": config.system_instruction,
            "roles": [c.role for c in contents],
            "texts": [c.parts[0].text for c in contents],
        })
        if self._usage_metadata == "default":
            usage = SimpleNamespace(prompt_token_count=self._usage[0], candidates_token_count=self._usage[1])
        else:
            usage = self._usage_metadata
        return SimpleNamespace(text=self._texts.pop(0), usage_metadata=usage)


def initial_state(message="what is my refund status", history=None):
    return {
        "session_id": "s", "business_id": "00000000-0000-0000-0000-000000000001",
        "user_message": message, "conversation_history": history or [],
        "intent": None, "retrieved_docs": [], "retrieved_doc_ids": [],
        "response_text": None, "safety_flag": None, "escalate": False,
        "latency_ms": 0, "token_count": 0,
    }


def run_graph(fake, state, docs=()):
    import asyncio

    async def go():
        with patch("app.agents.llm_client._client", fake), \
             patch("app.agents.graph.retrieve", new=AsyncMock(return_value=list(docs))):
            return await shrivox_agent.ainvoke(state)
    return asyncio.run(go())


# ------------------------------------------------------------ H1: system prompt

def test_every_system_message_is_kept_and_in_order():
    messages = [
        {"role": "system", "content": "FIRST persona and rules"},
        {"role": "system", "content": "SECOND stricter instruction"},
        {"role": "user", "content": "hello"},
    ]
    system_instruction, contents = _to_gemini_contents(messages)
    assert "FIRST persona and rules" in system_instruction
    assert "SECOND stricter instruction" in system_instruction
    assert system_instruction.index("FIRST") < system_instruction.index("SECOND")
    assert len(contents) == 1


def test_safety_retry_keeps_persona_rules_and_adds_the_stricter_instruction():
    """Ungrounded + 'guarantee' -> flagged -> one retry. The retry request must
    still carry the persona/rules AND the 'no relevant context' notice."""
    fake = FakeGemini(["general_query", "We guarantee a full refund.", "Let me check that for you."])

    out = run_graph(fake, initial_state("will I definitely get a refund?"), docs=[])

    assert len(fake.requests) == 3                      # intent, generation, ONE retry
    retry_system = fake.requests[2]["system_instruction"]
    assert "You are Shrivox" in retry_system            # persona and rules (SYSTEM_PROMPT)
    assert "Never make promises" in retry_system
    assert "No relevant context found." in retry_system
    assert "Do not make guarantees" in retry_system     # the stricter instruction
    assert out["response_text"] == "Let me check that for you."
    assert out["safety_flag"] is not None


# --------------------------------------------------------- H3: history roles

def test_history_roles_map_to_gemini_roles():
    history = [{"role": "customer", "content": "mera refund?"}, {"role": "agent", "content": "5-7 din lagenge."}]
    _, contents = _to_gemini_contents(build_response_prompt([], history, "aur kitna time?"))
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_openai_style_roles_still_work():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    _, contents = _to_gemini_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_unknown_history_role_is_dropped_and_logged_without_its_content():
    messages = [
        {"role": "tool", "content": "SENSITIVE-TOOL-OUTPUT"},
        {"role": "user", "content": "hello"},
    ]
    with capture_logs() as logs:
        _, contents = _to_gemini_contents(messages)
    assert [c.parts[0].text for c in contents] == ["hello"]
    events = [l for l in logs if l["event"] == "llm_history_role_skipped"]
    assert len(events) == 1 and "SENSITIVE-TOOL-OUTPUT" not in str(events)


def test_stored_conversation_history_reaches_gemini_with_correct_roles_through_the_graph():
    history = [{"role": "customer", "content": "mera order kahan hai"}, {"role": "agent", "content": "check kar raha hoon"}]
    fake = FakeGemini(["order_tracking", "Aapka order kal tak aa jayega."])

    run_graph(fake, initial_state("kab tak aayega", history=history),
              docs=[SimpleNamespace(text="Orders ship in 2 days.", id="c1")])

    assert fake.requests[1]["roles"] == ["user", "model", "user"]
    assert fake.requests[1]["texts"][-1] == "kab tak aayega"


# ------------------------------------------------------------ intent validation

def test_intent_prompt_and_vocabulary_cannot_drift_apart():
    from app.prompts.templates import DEFAULT_INTENT, VALID_INTENTS
    assert DEFAULT_INTENT in VALID_INTENTS
    assert "Categories: " + ", ".join(VALID_INTENTS) in INTENT_DETECTION_PROMPT
    assert "Categories: refund_request, order_tracking, payment_issue, account_issue, escalation_request, general_query, greeting, goodbye" in INTENT_DETECTION_PROMPT


@pytest.mark.parametrize("raw,expected", [
    ("refund_request", "refund_request"),
    ("  Order_Tracking\n", "order_tracking"),
    ("`general_query`", "general_query"),
    ("greeting.", "greeting"),
    ("**goodbye**", "goodbye"),
    ("Refund Request", "refund_request"),
    ("payment-issue", "payment_issue"),
    ("account_issue because they cannot log in", "account_issue"),
    ("escalation_request", "escalation_request"),      # preserved verbatim: escalation behaviour is unchanged here
    ("", "general_query"),
    ("   ", "general_query"),
    (None, "general_query"),
    ("I think this is probably a refund", "general_query"),
    ("refund_request; DROP TABLE calls", "refund_request"),   # only the vocabulary member is ever kept, never the raw text
    ("not_a_real_intent", "general_query"),
])
def test_normalize_intent(raw, expected):
    from app.agents.graph import normalize_intent
    assert normalize_intent(raw) == expected


def test_normalize_intent_always_returns_a_valid_short_intent_for_arbitrary_model_output():
    from app.agents.graph import normalize_intent
    from app.prompts.templates import VALID_INTENTS
    rng = random.Random(1234)                           # seeded: deterministic
    alphabet = string.ascii_letters + string.digits + string.punctuation + " \n\t" + "अआइईउ😀ünï"
    for _ in range(1000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        result = normalize_intent(raw)
        assert result in VALID_INTENTS and len(result) <= 50


def test_graph_never_stores_an_unbounded_or_unknown_intent():
    fake = FakeGemini(["The customer is asking about " + "x" * 300, "Sure, I can help."])
    out = run_graph(fake, initial_state("hello there"))
    assert out["intent"] == "general_query"
    assert len(out["intent"]) <= 50
    assert out["escalate"] is False


# ----------------------------------------------------------- usage-token guards

def test_missing_token_counts_become_zero_instead_of_crashing():
    fake = FakeGemini(["general_query", "Sure."], usage_metadata=SimpleNamespace(prompt_token_count=None, candidates_token_count=None))
    out = run_graph(fake, initial_state("hello"))
    assert out["token_count"] == 0


def test_missing_usage_metadata_entirely_does_not_crash():
    fake = FakeGemini(["general_query", "Sure."], usage_metadata=None)
    out = run_graph(fake, initial_state("hello"))
    assert out["token_count"] == 0


def test_reported_token_counts_are_still_summed():
    """prompt + completion tokens are summed per call, and now across EVERY call in
    the turn (classifier + generation): the classifier's spend used to go uncounted."""
    fake = FakeGemini(["general_query", "Sure."], usage=(12, 7))
    out = run_graph(fake, initial_state("hello"))
    assert out["token_count"] == 19 + 19
