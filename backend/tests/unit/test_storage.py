"""Storage frame tests — migrations are idempotent, WAL is on, locking holds."""
from __future__ import annotations

import threading

from kblocallm.storage.db import SQLiteManager
from kblocallm.storage.migrations import _MIGRATIONS, apply_migrations


def test_migrations_are_idempotent(tmp_path):
    """Runs on every boot. Must be safe to run twice.

    Derived from _MIGRATIONS rather than hardcoding the version: this test
    asserts idempotency, and pinning the literal count made it fail on every
    new migration for a reason that has nothing to do with what it checks.
    """
    db = SQLiteManager(tmp_path / "t.db")
    apply_migrations(db)
    apply_migrations(db)

    latest = max(version for version, _, _ in _MIGRATIONS)
    assert db.scalar("SELECT MAX(version) FROM schema_version") == latest
    # The real property: a second run applies nothing again.
    assert db.scalar("SELECT COUNT(*) FROM schema_version") == len(_MIGRATIONS)
    db.close()


def test_migration_versions_are_unique_and_ordered():
    """Append-only, one version each. A duplicate or out-of-order version
    means some databases silently skip a migration depending on the order they
    were last upgraded in — and HELM's DB holds operator data with no upstream
    source of truth to rebuild from."""
    versions = [version for version, _, _ in _MIGRATIONS]
    assert versions == sorted(set(versions))


def test_the_crash_detector_table_exists(tmp_path):
    """Migration 2. Without it the pid history has nowhere to live and every
    poll looks like a first sighting — i.e. no crash is ever detected."""
    db = SQLiteManager(tmp_path / "t.db")
    apply_migrations(db)
    columns = {r["name"] for r in db.execute("PRAGMA table_info(backend_state)")}
    assert {"model_name", "pid", "backend_alive", "last_restart_at"} <= columns
    db.close()


def test_wal_is_enabled(tmp_path):
    """WAL lets the dashboard poll read while an operator's tag write
    commits, instead of the two blocking each other."""
    db = SQLiteManager(tmp_path / "t.db")
    db.connect()
    assert db.scalar("PRAGMA journal_mode").lower() == "wal"
    db.close()


def test_foreign_keys_enforced(tmp_path):
    """sqlite3 leaves FKs OFF by default — verify we turned them on, or the
    model_tags -> model_metadata cascade silently does nothing."""
    db = SQLiteManager(tmp_path / "t.db")
    apply_migrations(db)
    assert db.scalar("PRAGMA foreign_keys") == 1
    db.close()


def test_tag_cascade_deletes(tmp_path):
    db = SQLiteManager(tmp_path / "t.db")
    apply_migrations(db)
    db.execute("INSERT INTO model_metadata (model_id) VALUES (?)", ("m1",))
    db.execute("INSERT INTO model_tags (model_id, tag) VALUES (?, ?)", ("m1", "hermes-preferred"))
    db.execute("DELETE FROM model_metadata WHERE model_id = ?", ("m1",))
    assert db.scalar("SELECT COUNT(*) FROM model_tags WHERE model_id = 'm1'") == 0
    db.close()


def test_concurrent_writes_do_not_corrupt(tmp_path):
    """FastAPI runs sync handlers on a threadpool, so the manager's RLock is
    load-bearing, not decoration."""
    db = SQLiteManager(tmp_path / "t.db")
    apply_migrations(db)

    def worker(n: int) -> None:
        for i in range(20):
            db.execute(
                "INSERT OR IGNORE INTO model_metadata (model_id) VALUES (?)",
                (f"model-{n}-{i}",),
            )

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert db.scalar("SELECT COUNT(*) FROM model_metadata") == 80
    db.close()
