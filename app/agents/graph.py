"""
The core conversation agent, built as a LangGraph StateGraph.

Why LangGraph instead of a plain linear function (input_safety ->
retrieve -> generate -> output_safety, called top to bottom):
Two of these steps are conditional, not linear. If input_safety flags
an escalation trigger, we must skip retrieval and generation entirely
and go straight to a "transfer to human" response -- running an LLM
call after already deciding to escalate wastes latency and money.
Similarly, if output_safety flags an ungrounded promise, the graph
should retry generation once with a stricter instruction rather than
silently sending a flagged response to TTS. A plain function can do
this with if/else, but as more conditional branches get added
(Phase 2+: barge-in handling, clarification loops, multi-intent
turns) the branching logic becomes unmanageable in nested if/else.
LangGraph makes each branch an explicit edge, which is the whole
reason for using it over a linear chain -- it's how you get "stateful
workflows" that resumt cleanly and read as decision-flow diagrams
rather than nested conditionals.

State shape below is intentionally a flat TypedDict (not a class with
methods) -- LangGraph passes state as plain dicts between nodes, so
each node is just "read what it needs off the dict, return the delta."
"""

import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.agents.llm_client import chat_completion
from app.agents.guardrails import check_input, check_output
from app.retrieval.retriever import retrieve
from app.prompts.templates import build_response_prompt, INTENT_DETECTION_PROMPT
from app.db.session import AsyncSessionLocal


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
    token_count: int


async def detect_intent(state: ConversationState) -> dict:
    input_check = check_input(state["user_message"])
    if not input_check.safe:
        return {"escalate": True, "safety_flag": input_check.flag, "intent": "escalation_request"}

    result = await chat_completion(
        [{"role": "user", "content": INTENT_DETECTION_PROMPT.format(message=state["user_message"])}],
        temperature=0.0,
        max_tokens=20,
    )
    return {"intent": result["text"].strip().lower(), "escalate": False}


async def retrieve_context(state: ConversationState) -> dict:
    if state.get("escalate"):
        return {"retrieved_docs": [], "retrieved_doc_ids": []}

    # Short-lived session scoped to this single retrieval call -- graph
    # state stays plain data (no long-lived DB session threaded through
    # it), and the business_id filter happens inside retrieve() itself,
    # in the SQL WHERE clause, not here in application code.
    async with AsyncSessionLocal() as db:
        docs = await retrieve(db, state["business_id"], state["user_message"], top_k=3)

    return {
        "retrieved_docs": [d.text for d in docs],
        "retrieved_doc_ids": [d.id for d in docs],
    }


async def generate_response(state: ConversationState) -> dict:
    if state.get("escalate"):
        return {
            "response_text": (
                "I understand -- let me connect you with a human agent who can help with this right away."
            ),
            "latency_ms": 0,
            "token_count": 0,
        }

    messages = build_response_prompt(
        state["retrieved_docs"], state["conversation_history"], state["user_message"]
    )
    result = await chat_completion(messages)
    return {
        "response_text": result["text"],
        "latency_ms": result["latency_ms"],
        "token_count": result["prompt_tokens"] + result["completion_tokens"],
    }


async def safety_check_output(state: ConversationState) -> dict:
    if state.get("escalate"):
        # LangGraph requires every node to write to at least one state
        # channel; returning {} here would raise InvalidUpdateError.
        # Re-affirming the existing flag is a no-op state-wise but
        # satisfies that contract.
        return {"safety_flag": state.get("safety_flag")}


    grounded = len(state["retrieved_docs"]) > 0
    output_check = check_output(state["response_text"], grounded_in_kb=grounded)
    if not output_check.safe:
        # One retry with a stricter instruction rather than silently
        # shipping a flagged response to TTS.
        safer_messages = build_response_prompt(
            state["retrieved_docs"], state["conversation_history"], state["user_message"]
        )
        safer_messages.insert(1, {
            "role": "system",
            "content": "Do not make guarantees or promises. State only what the context confirms.",
        })
        result = await chat_completion(safer_messages)
        return {"response_text": result["text"], "safety_flag": output_check.flag}

    return {"safety_flag": None}


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("safety_check_output", safety_check_output)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "safety_check_output")
    graph.add_edge("safety_check_output", END)

    return graph.compile()


shrivox_agent = build_graph()
