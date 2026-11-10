import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from app.agents import llm_client
from app.core import observability as obs


def _resp(text="hello", finish="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)],
                           usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4))


def _value(counter, *labels):
    return (counter.labels(*labels) if labels else counter)._value.get()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,expected", [("STOP", False), ("FinishReason.MAX_TOKENS", True), ("MAX_TOKENS", True), (None, False)])
async def test_truncation_is_detected_from_the_finish_reason(finish, expected):
    before = _value(obs.LLM_TRUNCATED)
    with patch.object(llm_client._client.aio.models, "generate_content", new=AsyncMock(return_value=_resp(finish=finish))):
        result = await llm_client.chat_completion([{"role": "user", "content": "hi"}])
    assert result["truncated"] is expected
    assert _value(obs.LLM_TRUNCATED) - before == (1 if expected else 0)


@pytest.mark.asyncio
async def test_a_response_without_candidates_is_not_called_truncated():
    resp = SimpleNamespace(text="ok", candidates=None, usage_metadata=None)
    with patch.object(llm_client._client.aio.models, "generate_content", new=AsyncMock(return_value=resp)):
        assert (await llm_client.chat_completion([{"role": "user", "content": "hi"}]))["truncated"] is False


@pytest.mark.asyncio
async def test_llm_calls_and_tokens_are_counted():
    calls, tokens = _value(obs.LLM_CALLS, "ok"), _value(obs.LLM_TOKENS, "completion")
    with patch.object(llm_client._client.aio.models, "generate_content", new=AsyncMock(return_value=_resp())):
        await llm_client.chat_completion([{"role": "user", "content": "hi"}])
    assert _value(obs.LLM_CALLS, "ok") - calls == 1 and _value(obs.LLM_TOKENS, "completion") - tokens == 4


@pytest.mark.asyncio
async def test_metrics_endpoint_uses_route_templates_not_concrete_paths():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.get("/health")
        await c.get("/no/such/path/8f3a-secret-looking-id")
        body = (await c.get("/metrics")).text
    assert 'route="/health"' in body and 'route="unmatched"' in body
    assert "secret-looking" not in body                  # unbounded/leaky label values never appear
    assert 'route="/metrics"' not in body                # scraping does not count itself


def test_log_level_is_honoured_and_output_is_json(capsys):
    obs.configure_logging("WARNING")
    log = structlog.get_logger()
    log.info("quiet_event")
    log.warning("loud_event", k=1)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and json.loads(out[0])["event"] == "loud_event"
    obs.configure_logging("not-a-level")                  # bad value falls back instead of crashing
    log.info("info_visible")
    assert "info_visible" in capsys.readouterr().out
    obs.configure_logging("INFO")
