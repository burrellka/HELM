"""SQLite connection management.

Shape borrowed from URSA-OSCAR's DuckDBManager (single connection + RLock +
materialized results), adapted to sqlite3. Same reasoning, different engine:

  - One writer. HELM is one container; nothing else opens this file.
  - Python's sqlite3 connection is not safe to share across threads without
    care, and FastAPI runs sync handlers on a threadpool. So every access
    goes through one RLock.
  - `execute()` takes the lock, runs, DRAINS the cursor into a list, then
    releases. Callers read results outside the lock without racing another
    in-flight query.
  - Multi-statement work (transactions, migrations) uses `serialized()` to
    hold the lock across the whole block.

Why SQLite and not DuckDB like URSA/Vitals: HELM stores operator metadata
and settings — small, transactional, key/value-ish. DuckDB is an analytics
engine; this is not an analytics workload. Deliberate divergence, recorded in
HELM_SPEC_REVIEW.md house-conflict #8.

WAL is on: it lets a read (the dashboard poll) proceed while a write (the
operator tagging a model) commits, without either blocking. On a bind-mounted
volume that's the difference between a snappy UI and a stuttering one.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class SQLiteManager:
    """One instance per process. Wired into app.state.db by main.lifespan."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None
        # RLock (not Lock) so one thread can nest serialized() -> execute().
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> sqlite3.Connection:
        """Open if needed. Idempotent. The returned connection is NOT safe to
        use without holding self._lock — go through execute()/serialized()."""
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._path),
                # We do our own locking; sqlite3's thread check would reject
                # the threadpool handing the connection between threads.
                check_same_thread=False,
                # Wait rather than instantly raising "database is locked" if
                # something else (a stray sqlite3 CLI) holds a write lock.
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL: concurrent reader + writer. Survives restarts (it's a
            # persistent property of the file, but setting it is cheap and
            # idempotent, and it documents the intent here).
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Durability/perf trade: NORMAL is the standard WAL pairing. A
            # power cut can lose the last transaction; for operator tags and
            # notes that's acceptable and it avoids an fsync per write.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # Not on by default in sqlite3. We want them.
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def serialized(self) -> Iterator[sqlite3.Connection]:
        """Hold the lock across a multi-statement block (transaction,
        migration, read-modify-write)."""
        with self._lock:
            yield self.connect()

    def execute(self, sql: str, params: tuple | list | None = None) -> list[sqlite3.Row]:
        """Run one statement, return fully-materialized rows.

        Returns a list, not a cursor, on purpose: by the time you read it the
        cursor is gone, so a concurrent execute() can't invalidate it.
        """
        with self._lock:
            conn = self.connect()
            cur = conn.execute(sql, tuple(params) if params else ())
            try:
                rows = cur.fetchall()
            finally:
                cur.close()
            conn.commit()
            return rows

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            conn = self.connect()
            conn.executemany(sql, rows)
            conn.commit()

    def query_one(self, sql: str, params: tuple | list | None = None) -> sqlite3.Row | None:
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: tuple | list | None = None) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row is not None else None
