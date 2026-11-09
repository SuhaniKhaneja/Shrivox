"""
The core conversation agent, built as a LangGraph StateGraph.

                         START
                           |
                      [input_guard]            deterministic rules, no I/O
                  escalate? |  no
              +-------------+--------------+
              |                            |  (fan-out: both start at once)
              |              [classify_intent] || [retrieve_context]
              |                            |  (join)
              |                        [triage]
              |              escalate? |  no
              +-------------+----------+
              |                         |
      [escalate_response]        [generate_response] <-----------+
              |                         |                        |
             END                  [safety_check]          [regenerate_response]
                                        |                        ^
                          safe -> END   |  unsafe, retries left -+
                                        |  unsafe, none left
                                  [safe_fallback] -> END (escalates)

Why a graph and not a linear function: the control flow is genuinely branching,
and each branch is an explicit edge you can read, draw and test.

  * The cheap deterministic guard runs FIRST and gates everything. A message that
    trips it never costs an LLM call, an embedding call or a database query.
  * Intent classification and retrieval do not depend on each other, so they run
    in parallel: the turn waits for the slower of the two, not their sum. The price
    is that when the classifier (not the guard) decides to escalate, the retrieval
    that ran alongside it was wasted -- one embedding call, in the rare case.
  * The output check is a loop with a bound. An unsafe answer is regenerated once
    with a stricter instruction and CHECKED AGAIN; if it is still unsafe it is
    never sent -- the customer gets a safe hand-off to a human instead.

State is a flat TypedDict: LangGraph passes plain dicts between nodes, so each node
reads what it needs and returns only the delta. `token_count` has an add-reducer so
every LLM call in the turn (classifier, generation, retry) contributes to it.
"""

import operator
import re
import uuid
from typing import Annotated, Literal, Optional, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from app.agents.guardrails import check_input, check_output
from app.agents.llm_client import chat_completion
from app.core.upstream import UpstreamEmptyResponse
from app.db.session import AsyncSessionLocal
from app.prompts.templates import DEFAULT_INTENT, INTENT_DETECTION_PROMPT, VALID_INTENTS, build_response_prompt
from app.retrieval.retriever import contextual_query, retrieve

logger = structlog.get_logger()

# An unsafe answer is regenerated this many times before falling back to a human.
MAX_SAFETY_RETRIES = 1

ESCALATION_MESSAGE = "I understand -- let me connect you with a human agent who can help with this right away."
SAFE_FALLBACK_MESSAGE = (
    "I want to be sure you get accurate information on this, so let me connect you "
    "with a human agent who can confirm the details."
)
STRICT_RETRY_INSTRUCTION = "Do not make guarantees or promises. State only what the context confirms."


class ConversationState(TypedDict):
    session_id: str
    business_id: uuid.UUID  # tenant scope -- see app/retrieval/retriever.py for why this must be threaded through
    user_message: str
    conversation_history: list[dict]  # prior turns, [{"role":..,"content":..}]
    intent: Optional[str]
    retrieved_docs: list[str]
    retrieved_doc_ids: list[str]
    response_text: Optional[str]
    safety_flag: Optional[str]
    escalate: bool
    latency_ms: int
    # Tokens spent by EVERY LLM call this turn. Nodes return their own call's count
    # and the reducer sums them (a bare int would keep only the last writer).
    token_count: Annotated[int, operator.add]
    # Set by the graph itself; callers need not supply them.
    safety_attempts: int
    output_safe: bool


_INTENT_PUNCTUATION = "`'\".,;:!?*()[]{}<>"


def normalize_intent(raw) -> str:
    """Map the classifier model's raw output onto VALID_INTENTS.

    Models decorate answers ("`greeting`", "Refund Request.", "refund_request
    because..."). The whole reply is tried first, then its first word; anything
    that still isn't in the vocabulary becomes DEFAULT_INTENT. The result is
    therefore ALWAYS a short vocabulary member, never raw model text -- which
    matters because it is persisted (Message.intent is String(50)) and used for
    analytics."""
    if not isinstance(raw, str) or not raw.strip():
        return DEFAULT_INTENT
    text = raw.strip()
    for candidate in (text, text.split(None, 1)[0]):
        cleaned = re.sub(r"[\s\-]+", "_", candidate.strip(_INTENT_PUNCTUATION).lower())
        if cleaned in VALID_INTENTS:
            return cleaned
    logger.warning("intent_out_of_vocabulary", raw_length=len(raw))  # length only, never the text
    return DEFAULT_INTENT


def _tokens(result: dict) -> int:
    return result["prompt_tokens"] + result["completion_tokens"]


# ------------------------------------------------------------------- nodes

async def input_guard(state: ConversationState) -> dict:
    """Deterministic input rules. No I/O, so it is free to run before anything else."""
    check = check_input(state["user_message"])
    if not check.safe:
        return {"escalate": True, "safety_flag": check.flag, "intent": "escalation_request"}
    return {"escalate": False}


async def classify_intent(state: ConversationState) -> dict:
    try:
        result = await chat_completion(
            [{"role": "user", "content": INTENT_DETECTION_PROMPT.format(message=state["user_message"])}],
            temperature=0.0,
            max_tokens=20,
        )
        return {"intent": normalize_intent(result["text"]), "token_count": _tokens(result)}
    except UpstreamEmptyResponse:
        # The classifier is best-effort: an empty reply must not fail the whole
        # turn. Other upstream failures (unavailable/rejected) still do.
        logger.warning("intent_empty_response")
        return {"intent": DEFAULT_INTENT}


async def retrieve_context(state: ConversationState) -> dict:
    # Short-lived session scoped to this single retrieval call -- graph
    # state stays plain data (no long-lived DB session threaded through
    # it), and the business_id filter happens inside retrieve() itself,
    # in the SQL WHERE clause, not here in application code.
    message = state["user_message"]
    async with AsyncSessionLocal() as db:
        docs = await retrieve(
            db, state["business_id"], message, top_k=3,
            # A short follow-up ("aur kitne din lagenge?") also borrows the topic of
            # the customer's previous message; the message itself is still searched.
            context_query=contextual_query(message, state["conversation_history"]),
        )

    return {
        "retrieved_docs": [d.text for d in docs],
        "retrieved_doc_ids": [d.id for d in docs],
    }


async def triage(state: ConversationState) -> dict:
    """The join point after the parallel branches. The classifier may have decided
    the customer is asking for a human; honour that. (Always writes `escalate`:
    LangGraph requires every node to write at least one state channel.)"""
    if state.get("intent") == "escalation_request":
        return {"escalate": True, "safety_flag": "escalation_intent"}
    return {"escalate": False}


async def escalate_response(state: ConversationState) -> dict:
    return {
        "response_text": ESCALATION_MESSAGE,
        "latency_ms": 0,
        # Retrieval may have run alongside the classifier, but none of it was used.
        "retrieved_docs": [],
        "retrieved_doc_ids": [],
    }


async def generate_response(state: ConversationState) -> dict:
    messages = build_response_prompt(
        state["retrieved_docs"], state["conversation_history"], state["user_message"]
    )
    result = await chat_completion(messages)
    return {
        "response_text": result["text"],
        "latency_ms": result["latency_ms"],
        "token_count": _tokens(result),
    }


async def safety_check(state: ConversationState) -> dict:
    grounded = len(state["retrieved_docs"]) > 0
    check = check_output(state["response_text"], grounded_in_kb=grounded)
    # Once the guardrail has intervened, the flag sticks even if the retry passes:
    # analytics need to see that it fired.
    return {"output_safe": check.safe, "safety_flag": check.flag or state.get("safety_flag")}


async def regenerate_response(state: ConversationState) -> dict:
    messages = build_response_prompt(
        state["retrieved_docs"], state["conversation_history"], state["user_message"]
    )
    messages.insert(1, {"role": "system", "content": STRICT_RETRY_INSTRUCTION})
    result = await chat_completion(messages)
    return {
        "response_text": result["text"],
        "latency_ms": state.get("latency_ms", 0) + result["latency_ms"],
        "token_count": _tokens(result),
        "safety_attempts": state.get("safety_attempts", 0) + 1,
    }


async def safe_fallback(state: ConversationState) -> dict:
    """The answer stayed unsafe after the bounded retry: do not send it."""
    logger.warning("output_blocked_after_retry", flag=state.get("safety_flag"),
                   attempts=state.get("safety_attempts", 0))
    return {"response_text": SAFE_FALLBACK_MESSAGE, "escalate": True}


# ----------------------------------------------------------------- routing

def route_after_guard(state: ConversationState) -> list[str] | str:
    if state.get("escalate"):
        return "escalate_response"
    return ["classify_intent", "retrieve_context"]  # fan-out: both run in the same step


def route_after_triage(state: ConversationState) -> Literal["escalate_response", "generate_response"]:
    return "escalate_response" if state.get("escalate") else "generate_response"


def route_after_safety(state: ConversationState) -> str:
    if state.get("output_safe"):
        return END
    if state.get("safety_attempts", 0) < MAX_SAFETY_RETRIES:
        return "regenerate_response"
    return "safe_fallback"


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("triage", triage)
    graph.add_node("escalate_response", escalate_response)
    graph.add_node("generate_response", generate_response)
    graph.add_node("safety_check", safety_check)
    graph.add_node("regenerate_response", regenerate_response)
    graph.add_node("safe_fallback", safe_fallback)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard,
                                ["escalate_response", "classify_intent", "retrieve_context"])
    # Join barrier: triage waits for BOTH parallel branches.
    graph.add_edge(["classify_intent", "retrieve_context"], "triage")
    graph.add_conditional_edges("triage", route_after_triage)
    graph.add_edge("escalate_response", END)
    graph.add_edge("generate_response", "safety_check")
    graph.add_conditional_edges("safety_check", route_after_safety,
                                [END, "regenerate_response", "safe_fallback"])
    graph.add_edge("regenerate_response", "safety_check")
    graph.add_edge("safe_fallback", END)

    return graph.compile()


shrivox_agent = build_graph()
