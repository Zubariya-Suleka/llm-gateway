"""Structured JSON logging on top of stdlib `logging`.

Usage: `log.info("request done", extra={"model": m, "latency_ms": 812})`.
Every key passed in `extra` becomes a top-level field in the JSON line.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else on the record came from `extra=`.
# `color_message` is uvicorn's ANSI-coloured duplicate of `msg`.
_SKIP_ATTRS = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _SKIP_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Send all logs (ours and uvicorn's) to stdout as JSON lines."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
