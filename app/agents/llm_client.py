"""
Thin wrapper around the Gemini client for reasoning (chat completion).

Why wrap it instead of calling the SDK directly from graph nodes:
Every LLM call needs the same three things applied consistently --
retry on transient failures, latency measurement (fed into analytics),
and a single place to swap models/providers later. Wrapping once
means graph nodes stay focused on orchestration logic, not on
"did this network call fail, should I retry."

Why Gemini 2.5 Flash instead of Groq's Llama 3.3 70B:
Unifies the whole stack on one provider (embeddings already use
Gemini) with one free-tier budget to track instead of two. Flash is
priced and benchmarked for exactly this job -- fast, cheap, high-volume
chat -- rather than reasoning-heavy tasks that would justify Pro.

Why this module still exposes an OpenAI-style `messages` list (roles:
system/user/assistant) instead of Gemini's native `contents` format:
The rest of the codebase (prompts/templates.py, agents/graph.py)
builds prompts in the familiar system/user/assistant shape. Isolating
the conversion to Gemini's format here means only this one file needs
to change if the provider changes again -- graph nodes never see a
provider-specific message format.
"""

import time
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types

from app.config import settings

logger = structlog.get_logger()

_client = genai.Client(api_key=settings.google_api_key)

# Roles in this codebase: prompts use OpenAI-style user/assistant, while stored
# conversation history (Redis) uses customer/agent. Both are mapped here, in the
# one place that talks Gemini's dialect ("user" / "model").
_USER_ROLES = {"user", "customer"}
_MODEL_ROLES = {"assistant", "agent", "model"}


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Splits an OpenAI-style message list into (system_instruction, contents).

    Gemini takes the system prompt as a separate field, not as a turn in the
    conversation. A request may carry SEVERAL system messages (the safety retry
    adds a stricter one after the persona/rules); they are all kept, joined in
    order -- keeping only the last would silently drop the persona and rules.

    Roles are mapped explicitly (see _USER_ROLES / _MODEL_ROLES). An unknown
    role is dropped and logged (role name only, never content) rather than
    guessed, so a corrupt history entry can't be sent as something the
    customer supposedly said."""
    system_parts: list[str] = []
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_parts.append(msg["content"])
            continue
        if role in _USER_ROLES:
            gemini_role = "user"
        elif role in _MODEL_ROLES:
            gemini_role = "model"
        else:
            logger.warning("llm_history_role_skipped", role=str(role)[:20])
            continue
        contents.append(types.Content(role=gemini_role, parts=[types.Part(text=msg["content"])]))
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
async def chat_completion(messages: list[dict], temperature: float = 0.3, max_tokens: int = 300) -> dict:
    system_instruction, contents = _to_gemini_contents(messages)

    start = time.perf_counter()

    resp = await _client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # The SDK reports usage fields as Optional; a missing value must not crash
    # the turn after the answer has already been generated. 0 means "not
    # reported", and thinking tokens are not included in completion_tokens.
    usage = resp.usage_metadata
    return {
        "text": resp.text,
        "latency_ms": latency_ms,
        "prompt_tokens": getattr(usage, "prompt_token_count", None) or 0,
        "completion_tokens": getattr(usage, "candidates_token_count", None) or 0,
    }