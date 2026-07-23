"""Phase 4 pure-logic tests: log classification (§5.6) and memory delta (F10)."""
from __future__ import annotations

import pytest

from kblocallm.domain.logs import LogEvent, classify
from kblocallm.domain.memory import Invalid, attribute


# -- log classification ----------------------------------------------------


def test_the_nuclear_option_is_not_read_as_routine_housekeeping():
    """THE ordering test. The mass-evict line also contains 'unloading', so a
    naive rule order reports a catastrophe as normal LRU behaviour. Every
    model just went cold; that must not look routine."""
    line = classify("2026-07-16 03:12:01 INFO Unloading models to clear hardware state")
    assert line.event is LogEvent.MASS_EVICT
    assert "went cold" in line.note
    assert "mid-incident" in line.note


def test_ordinary_eviction_is_reported_as_ordinary():
    line = classify("INFO evicting LRU model Aevum-0.6B to make room")
    assert line.event is LogEvent.EVICT
    assert "least recently used" in line.note


def test_watchdog_lines_are_caught_but_not_over_claimed():
    line = classify("2026-07-16 03:12:01 WARNING (BackendWatchdog) restarting backend pid 12345")
    assert line.event is LogEvent.WATCHDOG
    # F8: absence of the watchdog proves nothing, and the note must not imply
    # the operator can rely on it.
    assert "absence proves nothing" in line.note


def test_the_pool_line_is_the_sagas_tell():
    """'Largest memory pool:' is printed at every startup and went unread for
    months. A small value here is the whole 80GB saga in one line."""
    good = classify("llama_init: Largest memory pool: 120.50 GB")
    assert good.event is LogEvent.POOL
    assert "⚠" not in good.note

    bad = classify("llama_init: Largest memory pool: 0.50 GB")
    assert bad.event is LogEvent.POOL
    assert "⚠" in bad.note and "enable_dgpu_gtt" in bad.note


def test_slot_lines_are_ground_truth():
    """The backend states its own slot count here — the only place the real
    number comes from the thing doing the work rather than our parser."""
    line = classify("srv    init: initializing slots, n_slots = 8")
    assert line.event is LogEvent.SLOTS
    assert "8" in line.note


def test_new_slot_ctx_lines_are_caught():
    assert classify("slot         new slot n_ctx = 65536").event is LogEvent.SLOTS


def test_errors_are_flagged():
    assert classify("ERROR: failed to load model").event is LogEvent.ERROR


@pytest.mark.parametrize("line", ["", "   ", "INFO server listening on 0.0.0.0:13305"])
def test_ordinary_lines_are_plain_and_nothing_crashes(line):
    assert classify(line).event is LogEvent.PLAIN


def test_a_malformed_pool_value_does_not_crash():
    assert classify("Largest memory pool: not-a-number").event is not LogEvent.POOL


# -- memory attribution (F10) ----------------------------------------------


def _health(*names):
    return {"all_models_loaded": [{"model_name": n} for n in names]}


def _stats(vram):
    return {"vram_gb": vram}


def test_a_clean_delta_is_attributed_and_marked_estimated():
    est = attribute("new", _stats(20.0), _health("a"), _stats(34.5), _health("a", "new"))
    assert est.usable and est.estimated_gb == 14.5
    assert "estimate, not a measurement" in est.detail


def test_an_eviction_during_the_measurement_invalidates_it():
    """F10's core case: LRU eviction fires from the very load being measured,
    so the delta includes a negative contribution from whatever was dropped.
    Show nothing rather than a confidently wrong number."""
    est = attribute("new", _stats(100.0), _health("a", "b"), _stats(90.0), _health("a", "new"))
    assert not est.usable
    assert est.invalid is Invalid.COMPOSITION_CHANGED
    assert "b" in est.detail


def test_another_clients_concurrent_load_invalidates_it():
    """The CLI, KAIROS, or a second HELM tab loading at the same moment."""
    est = attribute("new", _stats(20.0), _health("a"), _stats(50.0), _health("a", "new", "someone-else"))
    assert est.invalid is Invalid.COMPOSITION_CHANGED
    assert "someone-else" in est.detail


def test_a_model_that_did_not_load_is_not_attributed():
    est = attribute("new", _stats(20.0), _health("a"), _stats(25.0), _health("a"))
    assert est.invalid is Invalid.TARGET_ABSENT


def test_a_model_already_loaded_yields_no_load_delta():
    est = attribute("new", _stats(20.0), _health("new"), _stats(25.0), _health("new"))
    assert est.invalid is Invalid.COMPOSITION_CHANGED


def test_a_negative_delta_is_refused_not_shown_as_zero():
    """VRAM going DOWN while a model loads means something else freed memory
    at the same moment. There is no honest number here."""
    est = attribute("new", _stats(50.0), _health("a"), _stats(40.0), _health("a", "new"))
    assert est.invalid is Invalid.IMPLAUSIBLE_DELTA
    assert est.estimated_gb is None


@pytest.mark.parametrize("bad", [None, {}, {"vram_gb": None}, {"vram_gb": "lots"}])
def test_a_missing_reading_is_refused(bad):
    est = attribute("new", bad, _health("a"), _stats(30.0), _health("a", "new"))
    assert est.invalid is Invalid.NO_READING
