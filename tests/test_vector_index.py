import pytest

from app.db.vector_index import supports_iterative_scan


@pytest.mark.parametrize("version,expected", [
    ("0.8.0", True), ("0.8.1", True), ("0.10.0", True), ("1.0.0", True),   # 0.10 > 0.8: compare numbers, not strings
    ("0.7.4", False), ("0.6.0", False), ("0.5.1", False),
    ("", False), (None, False), ("garbage", False),                          # unknown => do not assume support
])
def test_iterative_scan_support_is_detected_from_the_version(version, expected):
    assert supports_iterative_scan(version) is expected


@pytest.mark.asyncio
async def test_startup_does_not_wait_for_the_index_build():
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from app import main

    release = asyncio.Event()

    async def slow_build(_engine):
        await release.wait()            # stands in for a multi-minute CREATE INDEX
        return True

    @asynccontextmanager
    async def begin():
        yield MagicMock(execute=AsyncMock())

    with patch.object(main, "ensure_vector_index", new=slow_build), patch.object(main, "engine", new=SimpleNamespace(begin=begin)):
        await asyncio.wait_for(main.startup(), timeout=1)      # returns although the build is still running
        assert not main.app.state.vector_index_task.done()
        release.set()
        assert await main.app.state.vector_index_task is True
