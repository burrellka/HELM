"""Operator credential store — /data/auth.json. Ported from URSA-OSCAR.

Single-operator system. One password hash, a bootstrap flag, nothing else.
Deliberately a JSON file rather than a row in SQLite: it must be readable and
resettable by a human with a text editor when they've locked themselves out,
without needing a sqlite3 client on a Dockge box at midnight.

There is no password recovery. Losing it means deleting this file and
re-running setup. That's the correct trade for a single-user homelab tool and
it must be said out loud in the UI at setup time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AuthStore:
    def __init__(self, store_path: Path) -> None:
        self._path = Path(store_path)

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.exception(
                "auth.json at %s is unreadable (%s). Treating as NOT set up "
                "so the operator can recover via /setup.", self._path, e,
            )
            return {}

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a truncated
        # auth.json that locks the operator out.
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    @property
    def is_initialized(self) -> bool:
        return bool(self._read().get("password_hash"))

    def set_password_hash(self, password_hash: str) -> None:
        data = self._read()
        data["password_hash"] = password_hash
        self._write(data)

    def get_password_hash(self) -> str | None:
        return self._read().get("password_hash")
