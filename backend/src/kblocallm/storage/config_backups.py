"""Config backup history. The rail behind 14c §2.3 steps 5 and 7.

Every config apply snapshots the prior version here first, so an operator who
pastes a bad edit at 1am has a known-good version one click away. Backup
before write is not optional in the pipeline; it is the reason a raw editor is
safe to hand over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import SQLiteManager

#: Keep the last N per scope. A config that changes rarely does not need
#: unbounded history, and a runaway import loop should not fill the disk.
MAX_BACKUPS_PER_SCOPE = 50


@dataclass(frozen=True)
class Backup:
    id: int
    scope: str
    content: str
    reason: str
    created_at: str


class ConfigBackupStore:
    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def save(self, scope: str, content: str, reason: str = "apply") -> int:
        """Snapshot the PRIOR config text before a write. Returns the backup id.

        Called with the CURRENT (about-to-be-replaced) content, not the new
        one: the point is to preserve what was there so it can be restored.
        """
        with self._db.serialized() as conn:
            cur = conn.execute(
                "INSERT INTO config_backups (scope, content, reason) VALUES (?, ?, ?)",
                (scope, content, reason),
            )
            backup_id = cur.lastrowid
            # Trim old backups for this scope, keeping the most recent N.
            conn.execute(
                "DELETE FROM config_backups WHERE scope = ? AND id NOT IN ("
                "  SELECT id FROM config_backups WHERE scope = ? "
                "  ORDER BY created_at DESC, id DESC LIMIT ?"
                ")",
                (scope, scope, MAX_BACKUPS_PER_SCOPE),
            )
            conn.commit()
        return backup_id

    def list(self, scope: str, limit: int = 50) -> list[Backup]:
        rows = self._db.execute(
            "SELECT id, scope, content, reason, created_at FROM config_backups "
            "WHERE scope = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (scope, limit),
        )
        return [Backup(**dict(r)) for r in rows]

    def get(self, backup_id: int) -> Backup | None:
        row = self._db.query_one(
            "SELECT id, scope, content, reason, created_at FROM config_backups WHERE id = ?",
            (backup_id,),
        )
        return Backup(**dict(row)) if row else None
