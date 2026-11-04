"""
Test-suite-wide pytest configuration.

Integration tests (tests/integration) need real PostgreSQL with pgvector and
a real Redis, so they are opt-in:

    python -m pytest tests                        # unit suite; integration tests show as SKIPPED
    python -m pytest tests/integration --run-integration

Opting in and then lacking the infrastructure is a hard FAILURE, never a
silent skip -- a security test that quietly doesn't run is worse than none.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run tests that need real PostgreSQL (pgvector) and Redis",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ratelimit: exercise the real rate limiter (with a fake or real Redis)")
    config.addinivalue_line(
        "markers", "integration: needs real PostgreSQL (pgvector) + Redis; opt in with --run-integration"
    )


@pytest.fixture(autouse=True)
def _rate_limits_off_unless_requested(request, monkeypatch):
    """Unit tests have no Redis, and the limiter would otherwise wait on a dead
    connection for every login/register. Integration tests (real Redis) and tests
    marked `ratelimit` get the real limiter."""
    if "integration" in request.keywords or "ratelimit" in request.keywords:
        return
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_enabled", False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test: pass --run-integration (needs PostgreSQL+pgvector and Redis)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
