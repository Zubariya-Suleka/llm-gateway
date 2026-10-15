"""Response cache: deterministic (temperature 0) requests are answered from SQLite.

The key is the sha256 of the canonical JSON of every request field that can change the
answer. Leaving an input out would make two different requests share an answer; adding an
irrelevant one would make the cache never hit.
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from llm_gateway.db import connect

# Fields that never change what the model returns.
_IGNORED_FIELDS = frozenset({"stream", "user"})


def cache_key(material: dict[str, Any]) -> str:
    kept = {k: v for k, v in material.items() if k not in _IGNORED_FIELDS and v is not None}
    canonical = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_cacheable(temperature: float | None) -> bool:
    # Unset means the provider's default (1.0 on Groq), so only an explicit 0 is cached.
    return temperature == 0


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self._conn = connect(path)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row["body"]) if row else None

    def put(self, key: str, model: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, body, created_at)"
                " VALUES (?, ?, ?, ?)",
                (key, model, json.dumps(body), time.time()),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
