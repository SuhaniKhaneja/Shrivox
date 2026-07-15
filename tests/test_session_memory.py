import json
import pytest
from unittest.mock import AsyncMock, patch

from app.memory import session_memory


@pytest.fixture(autouse=True)
def reset_redis_singleton():
    session_memory._redis = None
    yield
    session_memory._redis = None


@pytest.mark.asyncio
async def test_append_turn_grows_history():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps([{"role": "user", "content": "hi"}])

    with patch("app.memory.session_memory.get_redis", return_value=mock_redis):
        history = await session_memory.append_turn("session-1", "assistant", "hello there")

    assert len(history) == 2
    assert history[-1] == {"role": "assistant", "content": "hello there"}
    mock_redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_history_capped_at_max_turns():
    from app.config import settings
    long_history = [{"role": "user", "content": f"msg {i}"} for i in range(settings.max_conversation_turns)]

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(long_history)

    with patch("app.memory.session_memory.get_redis", return_value=mock_redis):
        history = await session_memory.append_turn("session-1", "user", "one more message")

    assert len(history) == settings.max_conversation_turns
    assert history[-1]["content"] == "one more message"
    # oldest message should have been dropped
    assert history[0]["content"] != "msg 0"


@pytest.mark.asyncio
async def test_empty_history_returns_empty_list():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("app.memory.session_memory.get_redis", return_value=mock_redis):
        history = await session_memory.get_history("new-session")

    assert history == []
