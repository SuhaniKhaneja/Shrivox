"""
Metrics and logging setup.

Metrics answer "is it healthy right now and where is the time going"; logs answer
"what happened to this one request". Both are kept free of customer content:
labels are fixed vocabularies (route templates, intents, flags), never free text,
because an unbounded label value both leaks data and explodes Prometheus memory.

Counters live in the process: with several workers each exposes its own series and
the scraper sums them (standard Prometheus practice; no multiprocess mode needed
for counters and histograms that are scraped per instance).
"""

import logging
import time

import structlog
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS = Counter("shrivox_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("shrivox_http_request_seconds", "HTTP request latency", ["method", "route"],
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
LLM_CALLS = Counter("shrivox_llm_calls_total", "LLM calls by outcome", ["outcome"])
LLM_LATENCY = Histogram("shrivox_llm_call_seconds", "LLM call latency incl. retries",
                        buckets=(0.25, 0.5, 1, 2, 4, 8, 16))
LLM_TOKENS = Counter("shrivox_llm_tokens_total", "LLM tokens", ["kind"])
LLM_TRUNCATED = Counter("shrivox_llm_truncated_total", "Answers cut off by the output-token limit")
GUARD_EVENTS = Counter("shrivox_guard_events_total", "Guardrail interventions", ["stage", "flag"])
TURNS = Counter("shrivox_turns_total", "Completed chat turns", ["escalated"])


def route_label(request: Request) -> str:
    """The route TEMPLATE ('/conversations/{id}'), never the concrete path: ids in a
    label would be unbounded. Unmatched paths collapse into one bucket so scanners
    can't mint series either."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


async def metrics_middleware(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:  # an unhandled exception is still a request, counted as 500
        route = route_label(request)
        HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
        HTTP_LATENCY.labels(request.method, route).observe(time.perf_counter() - start)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def configure_logging(level: str = "INFO") -> None:
    """Make LOG_LEVEL actually do something, and emit one structured line per event."""
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        cache_logger_on_first_use=False,
    )
