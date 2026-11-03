"""Scriptable fakes at the Gemini SDK layer.

Patching the SDK client (not our wrapper functions) means the REAL
classification / retry / timeout / empty-response code runs in every test.
A script is a list of steps consumed one per SDK call:
  str        -> a successful reply with that text
  None       -> a reply whose text is None (what the SDK returns when blocked/empty)
  Exception  -> raised (build SDK errors with api_error())
  callable   -> awaited, its result is treated as a step again (use to sleep)
When the script runs out, sensible defaults apply so tests only script the interesting call.
"""

import hashlib
import math
import re
from types import SimpleNamespace

from google.genai import errors


def api_error(code: int, message: str = "upstream failure"):
    body = {"error": {"code": code, "message": message, "status": "ERR"}}
    return (errors.ServerError if code >= 500 else errors.ClientError)(code, body)


def default_vector(text: str, dim: int = 1536) -> list[float]:
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    if norm == 1.0 and not any(vec):
        vec[0] = 1.0
    return [x / norm for x in vec]


class ScriptedGemini:
    """Stands in for app.agents.llm_client._client."""

    def __init__(self, steps=None):
        self.steps = list(steps or [])
        self.requests: list[dict] = []
        self.attempts = 0
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model, contents, config):
        self.attempts += 1
        self.requests.append({
            "system_instruction": config.system_instruction,
            "roles": [c.role for c in contents],
            "texts": [c.parts[0].text for c in contents],
        })
        step = self.steps.pop(0) if self.steps else self._default(config)
        while callable(step):
            step = await step()
        if isinstance(step, BaseException):
            raise step
        return SimpleNamespace(
            text=step,
            usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=2),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    @staticmethod
    def _default(config):
        # The intent-classification call has no system instruction; answer generation does.
        return "general_query" if config.system_instruction is None else "Here is what I can tell you."


class ScriptedEmbedder:
    """Stands in for app.retrieval.retriever._genai_client."""

    def __init__(self, steps=None, vector_fn=default_vector):
        self.steps = list(steps or [])
        self.vector_fn = vector_fn
        self.calls: list[list[str]] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(embed_content=self._embed))

    async def _embed(self, model, contents, config):
        texts = list(contents) if isinstance(contents, list) else [contents]
        self.calls.append(texts)
        step = self.steps.pop(0) if self.steps else "ok"
        while callable(step):
            step = await step()
        if isinstance(step, BaseException):
            raise step
        if step == "short":                       # API returns fewer embeddings than inputs
            texts = texts[:-1]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=self.vector_fn(t)) for t in texts])
