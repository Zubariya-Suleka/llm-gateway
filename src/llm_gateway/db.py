"""SQLite connection shared by the cache and the metrics store."""

import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Calls arrive from worker threads (asyncio.to_thread); each store guards its
    # connection with a lock, so sharing it across threads is safe.
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL lets `gateway stats` read while the server writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
