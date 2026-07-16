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
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types

from app.config import settings

_client = genai.Client(api_key=settings.google_api_key)


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Splits an OpenAI-style message list into (system_instruction, contents).
    Gemini takes the system prompt as a separate field, not as a turn in
    the conversation, and uses role "model" where OpenAI-style APIs use
    "assistant"."""
    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
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

    usage = resp.usage_metadata
    return {
        "text": resp.text,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_token_count,
        "completion_tokens": usage.candidates_token_count,
    }