"""F8 tests: the pid is the detector, the watchdog only corroborates.

The scenario in `test_the_induced_crash_from_the_live_box` is the real one the
architect ran on the Lemonade host on 2026-07-16 — SIGKILL, pid 2628794 -> 2646871,
`watchdog_reset` false the whole way through. Any detector built on the flag
misses it entirely. That test is the reason this module exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kblocallm.domain.crash import (
    BackendObservation,
    BackendStatus,
    evaluate,
)
from kblocallm.storage.backends import BackendStateStore
from kblocallm.storage.db import SQLiteManager
from kblocallm.storage.migrations import apply_migrations

T0 = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _model(pid=1000, alive=True, status="ready", watchdog=False, name="Aevum-0.6B"):
    return {
        "model_name": name, "pid": pid, "backend_alive": alive,
        "status": status, "watchdog_reset": watchdog,
    }


def _seen(pid=1000, alive=True, name="Aevum-0.6B", **kw):
    return BackendObservation(model_name=name, pid=pid, backend_alive=alive, **kw)


# -- the detector ----------------------------------------------------------


def test_the_induced_crash_from_the_live_box():
    """The exact live scenario. A watchdog_reset-based detector reports
    nothing here; the pid comparison catches it."""
    state, _ = evaluate(_model(pid=2646871, watchdog=False), _seen(pid=2628794), now=T0)
    assert state.restarted_now is True
    assert state.last_restart_from == 2628794
    assert state.last_restart_at == T0
    assert "2628794 to 2646871" in state.detail


def test_watchdog_false_does_not_suppress_the_finding():
    """The flag being false must not be read as 'no crash'. It stays false
    when a request lazily reloads a dead backend first — the normal case on a
    busy model."""
    state, _ = evaluate(_model(pid=2, watchdog=False), _seen(pid=1), now=T0)
    assert state.restarted_now is True
    assert "does NOT rule out a crash" in state.detail


def test_watchdog_true_upgrades_the_claim_to_a_confirmed_crash():
    """Corroboration is additive: when the flag IS set, the watchdog did the
    restarting, and that does mean a crash."""
    state, _ = evaluate(_model(pid=2, watchdog=True), _seen(pid=1), now=T0)
    assert state.restarted_now and state.watchdog_reset
    assert "confirms a crash" in state.detail


def test_a_stable_pid_is_not_a_restart():
    state, _ = evaluate(_model(pid=1000), _seen(pid=1000), now=T0)
    assert state.restarted_now is False
    assert state.status is BackendStatus.READY


def test_watchdog_true_alone_is_not_reported_as_a_restart_now():
    """The flag is sticky-ish and says 'ever', not 'just now'. Treating it as
    a fresh event would fire on every poll forever — the permanent warning
    nobody reads."""
    state, _ = evaluate(_model(pid=1000, watchdog=True), _seen(pid=1000), now=T0)
    assert state.restarted_now is False
    assert state.watchdog_reset is True  # still surfaced, just not as news


def test_first_sighting_is_never_a_restart():
    """Otherwise every model cries wolf on HELM's first poll after a container
    start, and the operator learns to ignore the signal on day one."""
    state, _ = evaluate(_model(pid=1000), None, now=T0)
    assert state.restarted_now is False
    assert state.last_restart_at is None


@pytest.mark.parametrize("bad", [None, 0, -1, "1000", True])
def test_an_unknown_pid_never_manufactures_a_restart(bad):
    """Unknown != changed. Comparing None to None (or to a real pid) must not
    invent an incident."""
    state, _ = evaluate(_model(pid=bad), _seen(pid=None), now=T0)
    assert state.restarted_now is False
    state, _ = evaluate(_model(pid=bad), _seen(pid=1000), now=T0)
    assert state.restarted_now is False


# -- status ----------------------------------------------------------------


def test_backend_alive_false_is_dead():
    state, _ = evaluate(_model(alive=False), _seen(), now=T0)
    assert state.status is BackendStatus.DEAD
    assert "requests to it will fail" in state.detail


def test_the_alive_to_dead_transition_is_recorded_without_a_new_pid():
    """A process that dies and is NOT replaced has no new pid to compare. The
    pid stays stable and it looks healthy right up until a request fails."""
    _, obs = evaluate(_model(pid=1000, alive=False), _seen(pid=1000, alive=True), now=T0)
    assert obs.last_restart_at == T0
    assert obs.backend_alive is False


def test_a_non_ready_status_is_not_silently_ready():
    state, _ = evaluate(_model(status="loading"), _seen(), now=T0)
    assert state.status is BackendStatus.NOT_READY
    assert "loading" in state.detail


def test_restart_recency_ages_out():
    """F8's real objection: a permanent warning trains the operator to ignore
    warnings. Recency is what lets the UI be loud now and quiet later without
    ever hiding the fact."""
    state, _ = evaluate(_model(pid=2), _seen(pid=1), now=T0)
    assert state.is_recent_restart(T0 + timedelta(minutes=5)) is True
    assert state.is_recent_restart(T0 + timedelta(hours=3)) is False


# -- persistence -----------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = SQLiteManager(tmp_path / "helm.db")
    apply_migrations(db)
    yield BackendStateStore(db)
    db.close()


def test_a_restart_is_detected_across_polls(store):
    """The end-to-end shape: poll, restart, poll. Nothing in Lemonade's
    payload remembers the old pid, so this only works because HELM does."""
    assert store.observe_all([_model(pid=1000)])["Aevum-0.6B"].restarted_now is False
    states = store.observe_all([_model(pid=2000)])
    assert states["Aevum-0.6B"].restarted_now is True
    assert states["Aevum-0.6B"].last_restart_from == 1000


def test_the_restart_persists_after_it_stops_being_news(store):
    """The pid is stable again on the next poll, but the incident still
    happened and the row must keep saying so — with its timestamp."""
    store.observe_all([_model(pid=1000)], now=T0)
    store.observe_all([_model(pid=2000)], now=T0)
    later = store.observe_all([_model(pid=2000)], now=T0 + timedelta(minutes=30))
    state = later["Aevum-0.6B"]
    assert state.restarted_now is False        # not news any more
    assert state.last_restart_at == T0         # but not forgotten
    assert state.is_recent_restart(T0 + timedelta(minutes=30)) is False


def test_history_survives_an_unload_reload_gap(store):
    """Rows for absent models are kept. If they were deleted, the first poll
    after a reload would look like a fresh sighting and silently swallow a
    crash that happened across the gap."""
    store.observe_all([_model(pid=1000)], now=T0)
    store.observe_all([], now=T0)                     # model unloaded
    states = store.observe_all([_model(pid=3000)], now=T0)
    assert states["Aevum-0.6B"].restarted_now is True
    assert states["Aevum-0.6B"].last_restart_from == 1000


def test_multiple_models_are_tracked_independently(store):
    store.observe_all([_model(pid=1, name="a"), _model(pid=2, name="b")], now=T0)
    states = store.observe_all([_model(pid=1, name="a"), _model(pid=99, name="b")], now=T0)
    assert states["a"].restarted_now is False
    assert states["b"].restarted_now is True


def test_the_live_health_fixture_produces_no_false_restarts(store):
    """Three real loaded models, polled twice, nothing changed. A detector
    that flags anything here is unusable — it would fire every 5 seconds."""
    import json
    from pathlib import Path
    payload = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "health.json").read_text()
    )
    models = payload["all_models_loaded"]
    store.observe_all(models, now=T0)
    states = store.observe_all(models, now=T0)
    assert len(states) == 3
    assert not any(s.restarted_now for s in states.values())
    assert all(s.status is BackendStatus.READY for s in states.values())
