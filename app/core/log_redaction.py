"""
Scrub credentials out of server access logs.

Browsers cannot send an Authorization header on a WebSocket handshake, so the
chat ticket travels as `?token=...`. uvicorn logs the full request path
including the query string -- HTTP requests via `uvicorn.access`, and
WebSocket lines ('"WebSocket /path?token=..." [accepted]' / 403) via
`uvicorn.error`. This filter rewrites `token=<value>` to `token=[REDACTED]` in
the record before any handler formats it.

This is defense in depth, not a substitute for a short ticket lifetime:
reverse proxies, load balancers and browser history sit outside this process
and may still record the URL.
"""

import logging
import re

_TOKEN_QS = re.compile(r"([?&]token=)[^&\s\"']+", re.IGNORECASE)


def redact_token(text: str) -> str:
    return _TOKEN_QS.sub(r"\1[REDACTED]", text)


class RedactTokenFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_token(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_token(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact_token(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def install_token_redaction() -> None:
    """Idempotent. Attach to the loggers uvicorn uses for request lines."""
    for name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, RedactTokenFilter) for f in lg.filters):
            lg.addFilter(RedactTokenFilter())
