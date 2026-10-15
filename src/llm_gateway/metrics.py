"""One row per request in SQLite table `requests`.

latency_ms is the duration of the final provider call (or the cache lookup on a hit): the
model's latency. total_ms is wall time inside the gateway, including rate-limit waits and
retries: what the caller felt. Keeping both stops our own throttling from polluting the
model latency numbers.
"""

import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_gateway.db import connect


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="milliseconds")


@dataclass
class RequestRecord:
    model: str
    project: str | None = None
    trace_id: str | None = None
    provider: str | None = None
    ts: str = field(default_factory=utc_iso)
    status: int = 0
    cache_hit: bool = False
    attempts: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    total_ms: float | None = None
    completion_time_s: float | None = None
    tokens_per_sec: float | None = None
    est_cost_usd: float | None = None
    error: str | None = None


_COLUMNS = [f.name for f in fields(RequestRecord)]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    project TEXT,
    trace_id TEXT,
    provider TEXT,
    model TEXT NOT NULL,
    status INTEGER NOT NULL,
    cache_hit INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms REAL,
    total_ms REAL,
    completion_time_s REAL,
    tokens_per_sec REAL,
    est_cost_usd REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS requests_model_ts ON requests (model, ts);
"""


class MetricsStore:
    def __init__(self, path: Path) -> None:
        self._conn = connect(path)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    def record(self, rec: RequestRecord) -> None:
        values = asdict(rec)
        values["cache_hit"] = int(rec.cache_hit)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO requests ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                [values[c] for c in _COLUMNS],
            )

    def query(
        self,
        *,
        since: datetime | None = None,
        project: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(utc_iso(since))
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM requests {where} ORDER BY id", params
            ).fetchall()
        return [dict(r) for r in rows]

    def usage_since(self, model: str, since: datetime) -> tuple[int, int]:
        """(provider calls, tokens) for a model since `since`; cache hits cost nothing."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COALESCE(SUM(attempts), 0),
                          COALESCE(SUM(COALESCE(prompt_tokens, 0)
                                       + COALESCE(completion_tokens, 0)), 0)
                   FROM requests WHERE model = ? AND ts >= ? AND cache_hit = 0""",
                (model, utc_iso(since)),
            ).fetchone()
        return int(row[0]), int(row[1])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
