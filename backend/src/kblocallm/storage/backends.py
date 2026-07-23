"""Persistence for the crash detector. Pairs with domain/crash.py.

The split is deliberate: `domain.crash.evaluate` is pure and table-tested,
this file does the I/O and owns nothing but reads and writes. All the
judgement lives in the tested half.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..domain.crash import BackendObservation, BackendState, evaluate
from .db import SQLiteManager

logger = logging.getLogger(__name__)


class BackendStateStore:
    """Last-seen pid per model, and the restart history derived from it."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def observe_all(
        self,
        models: list[dict[str, Any]],
        now: datetime | None = None,
    ) -> dict[str, BackendState]:
        """Compare this poll's health records against the last one.

        Returns state keyed by model_name. Called on every dashboard poll, so
        the whole thing is one transaction: a torn read here would compare
        some models against this poll and others against the last, and
        manufacture restarts that never happened.

        Rows for models that are no longer loaded are KEPT on purpose. An
        unload followed by a reload is not a restart, but the pid history is
        still the truth about that backend, and deleting it would make the
        first poll after a reload look like a fresh sighting (silently
        suppressing a real crash that happened across the gap).
        """
        now = now or datetime.now(timezone.utc)
        states: dict[str, BackendState] = {}

        with self._db.serialized() as conn:
            previous = self._load_all(conn)
            for model in models:
                name = model.get("model_name")
                if not name:
                    # No key, no history. Still evaluate it so the row renders
                    # a status; just don't try to persist it under "".
                    state, _ = evaluate(model, None, now=now)
                    continue
                state, observation = evaluate(model, previous.get(name), now=now)
                states[name] = state
                self._upsert(conn, observation, now)
                if state.restarted_now:
                    # Worth a log line on its own: the dashboard may not be
                    # open when this happens, and this is the record that it
                    # did.
                    logger.warning(
                        "Backend restart detected for %s: pid %s -> %s (watchdog_reset=%s)",
                        name, state.last_restart_from, state.pid, state.watchdog_reset,
                    )
            conn.commit()

        return states

    # -- internals ---------------------------------------------------------

    def _load_all(self, conn) -> dict[str, BackendObservation]:
        rows = conn.execute(
            "SELECT model_name, pid, backend_alive, last_restart_at, "
            "       last_restart_from, last_restart_watchdog "
            "FROM backend_state"
        ).fetchall()
        return {
            row["model_name"]: BackendObservation(
                model_name=row["model_name"],
                pid=row["pid"],
                backend_alive=bool(row["backend_alive"]),
                last_restart_at=_parse_ts(row["last_restart_at"]),
                last_restart_from=row["last_restart_from"],
                last_restart_watchdog=bool(row["last_restart_watchdog"]),
            )
            for row in rows
        }

    def _upsert(self, conn, obs: BackendObservation, now: datetime) -> None:
        conn.execute(
            """
            INSERT INTO backend_state (
                model_name, pid, backend_alive, last_restart_at,
                last_restart_from, last_restart_watchdog, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                pid                   = excluded.pid,
                backend_alive         = excluded.backend_alive,
                last_restart_at       = excluded.last_restart_at,
                last_restart_from     = excluded.last_restart_from,
                last_restart_watchdog = excluded.last_restart_watchdog,
                updated_at            = excluded.updated_at
            """,
            (
                obs.model_name,
                obs.pid,
                1 if obs.backend_alive else 0,
                obs.last_restart_at.isoformat() if obs.last_restart_at else None,
                obs.last_restart_from,
                1 if obs.last_restart_watchdog else 0,
                now.isoformat(),
            ),
        )


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        # A timestamp we can't read is not worth crashing a dashboard poll
        # over. Losing the "restarted 3m ago" annotation degrades the display;
        # raising here would blank the whole table.
        logger.warning("Unparseable timestamp in backend_state: %r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
