"""Backend liveness + restart detection. Spec Rule 8, review F8, handoff §3.

**`watchdog_reset` is not the crash detector.** It reads like one, and that is
the trap. The architect induced a real crash on the live box (SIGKILL of
Aevum's backend, pid 2628794 -> 2646871) and `watchdog_reset` stayed `false`
throughout: the next request lazily reloaded the backend before the watchdog
ever polled it, so the watchdog was never the component that reset anything.
On a busy model that is the NORMAL case, which makes the flag reliable only
for crashes nobody noticed.

So the detector is the **pid**. Each loaded backend is its own OS process
(spec §3, hub-and-spoke); if the pid under a model changes, that process died
and a new one took its place. HELM persists the last-seen pid and compares.
`watchdog_reset: true` corroborates when present; it never leads.

**Wording matters here.** A pid change proves a restart. It does NOT prove a
crash: someone at the CLI running `lemonade load` produces the same evidence.
Phase 1 issues no loads of its own, so a change is always something HELM did
not do — but "restarted" is what we can actually see, and that is what the UI
says. Claiming "CRASHED" on the operator's own deliberate reload is the kind
of confident wrongness this product exists to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

#: A restart older than this is history, not news: shown with its timestamp,
#: but not shouted about. F8's objection to a raw sticky flag was that a
#: permanent warning teaches the operator to ignore warnings, which costs you
#: the NEXT incident. A timestamped fact ages itself out of urgency honestly.
RECENT_RESTART_WINDOW = timedelta(minutes=15)


class BackendStatus(str, Enum):
    READY = "ready"
    #: backend_alive is false. The router reaped the process and has not
    #: replaced it. Distinct from "unloaded" — the model is still listed.
    DEAD = "dead"
    #: Loaded, but the backend hasn't reported ready (starting, or degraded).
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class BackendObservation:
    """What HELM last saw for one model. Persisted in HELM's SQLite."""

    model_name: str
    pid: int | None
    backend_alive: bool
    last_restart_at: datetime | None = None
    last_restart_from: int | None = None
    last_restart_watchdog: bool = False


@dataclass(frozen=True)
class BackendState:
    """The evaluated per-model state the dashboard renders."""

    status: BackendStatus
    #: Raw `status` string from the router, passed through for display.
    raw_status: str | None = None
    pid: int | None = None
    #: True when the pid changed between this poll and the last one.
    restarted_now: bool = False
    last_restart_at: datetime | None = None
    last_restart_from: int | None = None
    #: `watchdog_reset` from the health record. Corroborating evidence only;
    #: it is FALSE after real crashes (see the module docstring), so it can
    #: raise confidence but must never be required.
    watchdog_reset: bool = False
    detail: str | None = None

    def is_recent_restart(self, now: datetime) -> bool:
        if self.last_restart_at is None:
            return False
        return (now - self.last_restart_at) <= RECENT_RESTART_WINDOW


def _pid(model: dict[str, Any]) -> int | None:
    pid = model.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return pid


def evaluate(
    model: dict[str, Any],
    previous: BackendObservation | None,
    now: datetime | None = None,
) -> tuple[BackendState, BackendObservation]:
    """Compare one health record against what we last saw.

    Returns the state to render and the observation to persist. Pure: the
    caller owns both the clock and the database, so this is table-testable.
    """
    now = now or datetime.now(timezone.utc)
    name = model.get("model_name") or ""
    pid = _pid(model)
    alive = model.get("backend_alive") is not False
    watchdog = model.get("watchdog_reset") is True
    raw_status = model.get("status")

    restarted_now = False
    restart_at = previous.last_restart_at if previous else None
    restart_from = previous.last_restart_from if previous else None
    restart_watchdog = previous.last_restart_watchdog if previous else False

    # A pid change is the signal. Both sides must be known: the first sighting
    # of a model (previous is None, or previous.pid is None) is not a restart,
    # and reporting one on HELM's own first poll would cry wolf at every
    # container start.
    if previous is not None and previous.pid is not None and pid is not None:
        if previous.pid != pid:
            restarted_now = True
            restart_at, restart_from, restart_watchdog = now, previous.pid, watchdog

    # backend_alive true -> false is the other transition worth catching: the
    # process died and has NOT been replaced, so there is no new pid to
    # compare against. Without this, a dead backend that never respawns shows
    # a stable pid and looks perfectly healthy right up until a request fails.
    if previous is not None and previous.backend_alive and not alive:
        restart_at, restart_from, restart_watchdog = now, previous.pid, watchdog

    if not alive:
        status = BackendStatus.DEAD
        detail = (
            "The router reports this backend's process is not alive. The model "
            "is still listed as loaded, but requests to it will fail."
        )
    elif raw_status == "ready":
        status, detail = BackendStatus.READY, None
    else:
        status = BackendStatus.NOT_READY
        detail = f"Loaded, but the backend reports status {raw_status!r} rather than ready."

    if restarted_now:
        detail = _restart_detail(restart_from, pid, watchdog)

    state = BackendState(
        status=status,
        raw_status=raw_status,
        pid=pid,
        restarted_now=restarted_now,
        last_restart_at=restart_at,
        last_restart_from=restart_from,
        watchdog_reset=watchdog,
        detail=detail,
    )
    observation = BackendObservation(
        model_name=name,
        pid=pid,
        backend_alive=alive,
        last_restart_at=restart_at,
        last_restart_from=restart_from,
        last_restart_watchdog=restart_watchdog,
    )
    return state, observation


def _restart_detail(old_pid: int | None, new_pid: int | None, watchdog: bool) -> str:
    """Say what we saw, not what we assume.

    A pid change is evidence of a restart, full stop. Whether it was a crash
    or someone at the CLI is not something HELM can see, so it does not claim
    to. The watchdog clause is additive: when the flag IS set it tells us the
    watchdog did the restarting, which does mean a crash.
    """
    base = (
        f"The backend process changed from pid {old_pid} to {new_pid}, so it "
        f"restarted since the last poll. HELM did not do this — it was a crash "
        f"or a change made outside HELM."
    )
    if watchdog:
        return base + " watchdog_reset is set, which confirms a crash: the router's watchdog restarted it."
    return base + (
        " watchdog_reset is not set, which does NOT rule out a crash — the flag "
        "stays false when a request lazily reloads a dead backend before the "
        "watchdog notices."
    )
