"""Assembly tests, driven off the four live captures from the Lemonade host.

These assert the numbers the operator will actually read on 2026-07-16's fleet
state. The `_live` fixture is the whole Phase 1 payload end to end, minus the
HTTP.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kblocallm.domain.dashboard import (
    PLAUSIBLE_MAX_TOKENS_PER_SECOND,
    SectionError,
    build_dashboard,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def live():
    return build_dashboard(
        health=_fixture("health"),
        system_stats=_fixture("system-stats"),
        system_info=_fixture("system-info"),
        stats=_fixture("stats"),
        params=_fixture("params"),
        management_enabled=True,
        now=NOW,
    )


def _row(dash, name):
    return next(r for r in dash.models if r.model_name.startswith(name))


# -- the resource header ---------------------------------------------------


def test_the_bar_is_drawn_against_the_real_pool(live):
    pool = live.resources.pool
    assert pool.capacity_gb == 120.0
    assert pool.used_gb == pytest.approx(56.42, abs=0.01)
    assert pool.percent == pytest.approx(47.0, abs=0.5)
    assert pool.warning is None
    assert pool.source == "system-info.devices.amd_gpu[0].virtual_mem_gb"


def test_the_gtt_flag_is_read_from_params_and_cross_checked(live):
    """The saga's cross-check needs enable_dgpu_gtt, which is not in
    system-info. /api/v1/params serves it, so HELM can do the check with no
    host filesystem access at all."""
    assert live.resources.pool.gtt_enabled is True


def test_npu_percent_stays_none_and_is_never_coerced_to_zero(live):
    """Rule 3. The gauge is broken upstream and always null. A 0.0 here is the
    lie; None is what lets the frontend render 'unavailable' honestly."""
    assert live.resources.npu_percent is None


def test_a_real_zero_gauge_would_survive(live):
    """The other half of Rule 3. If the gauge is ever fixed and legitimately
    reports 0, that 0 must reach the UI as a number — otherwise we replace
    the old lie with a new one in the same slot."""
    dash = build_dashboard(
        health={}, system_stats={"npu_percent": 0.0, "cpu_percent": 0.0},
        system_info=None, stats=None, params=None, now=NOW,
    )
    assert dash.resources.npu_percent == 0.0
    assert dash.resources.cpu_percent == 0.0


def test_gpu_and_cpu_come_through(live):
    assert live.resources.gpu_percent == 100.0
    assert live.resources.cpu_percent == pytest.approx(0.31, abs=0.01)
    assert live.resources.system_ram_gb == 66.0


def test_capacity_is_per_type_never_a_global_cap(live):
    """Rule 6. Three LLMs of a ten-LLM ceiling — and that ceiling is the LLM
    ceiling, not the box's."""
    by_type = {c.type: c for c in live.resources.capacity}
    assert (by_type["llm"].loaded, by_type["llm"].max, by_type["llm"].pinned) == (3, 10, 2)
    # Every other type publishes its own separate ceiling of 10.
    assert by_type["embedding"].loaded == 0 and by_type["embedding"].max == 10
    assert len(by_type) == 6


# -- the models table ------------------------------------------------------


def test_the_three_live_models_are_listed(live):
    assert len(live.models) == 3
    assert live.lemonade_version == "10.10.0"


def test_the_26b_row_reads_correctly(live):
    """The row six agents and three humans depend on."""
    row = _row(live, "Gemma-4-26B")
    assert row.concurrency.display == "8 slots × 64K"
    assert row.backend_label == "llamacpp/rocm"
    assert row.device == "gpu"
    assert row.pinned is True
    assert row.status == "ready"
    assert (row.pid, row.port) == (1771790, 8001)


def test_aevum_is_the_next_to_evict_and_the_pinned_pair_are_exempt(live):
    """Aevum is the only unpinned model, so it is what the router drops under
    pressure — even though its last_use is the most RECENT of the three. A
    naive 'oldest gets evicted' would have condemned the pinned 26B instead."""
    assert _row(live, "Aevum").next_to_evict is True
    assert _row(live, "Gemma-4-26B").next_to_evict is False
    assert _row(live, "Hermes-3").next_to_evict is False


def test_lru_rank_orders_by_last_use(live):
    """Rank, not '3m ago'. last_use is a monotonic counter with no epoch and
    no current-value endpoint to difference against, so a duration would be
    fabricated. Rank is what the LRU column is for anyway."""
    assert _row(live, "Aevum").lru_rank == 1        # 375540865, most recent
    assert _row(live, "Gemma-4-26B").lru_rank == 2  # 289649785
    assert _row(live, "Hermes-3").lru_rank == 3     # 288285642, least recent
    assert _row(live, "Aevum").last_use == 375540865


def test_no_row_displays_a_raw_ctx_size(live):
    """Rule 1's negative form, checked at the assembly layer too."""
    for row in live.models:
        assert "524288" not in row.concurrency.display
        assert "131072" not in row.concurrency.display


# -- throughput ------------------------------------------------------------


def test_the_implausible_rate_from_the_live_box_is_flagged_not_shown_as_fact(live):
    """The live box reports exactly 1,000,000.0 tok/s: a 1-token response
    divided by a near-zero elapsed. Printing that next to 'tok/s' would be the
    dashboard's first confident lie."""
    assert live.throughput.tokens_per_second == 1_000_000.0
    assert live.throughput.rate_implausible is True
    assert "not a real rate" in live.throughput.rate_note
    assert "1 token" in live.throughput.rate_note


def test_a_normal_rate_is_not_flagged():
    dash = build_dashboard(
        health={}, system_stats=None, system_info=None,
        stats={"tokens_per_second": 42.5, "output_tokens": 300}, params=None, now=NOW,
    )
    assert dash.throughput.rate_implausible is False
    assert dash.throughput.rate_note is None
    assert PLAUSIBLE_MAX_TOKENS_PER_SECOND == 10_000.0


def test_lifetime_totals_come_through(live):
    assert live.throughput.request_count_total == 18
    assert live.throughput.time_to_first_token == pytest.approx(0.02541)


# -- degradation -----------------------------------------------------------


def test_one_dead_read_does_not_blank_the_others():
    """Partial degradation. /stats failing must not cost the operator the
    models table — they are usually looking at HELM because something is
    already wrong."""
    dash = build_dashboard(
        health=_fixture("health"),
        system_stats=_fixture("system-stats"),
        system_info=_fixture("system-info"),
        stats=None,
        params=_fixture("params"),
        errors={"stats": SectionError(kind="timeout", detail="ReadTimeout")},
        now=NOW,
    )
    assert dash.throughput is None
    assert len(dash.models) == 3                     # still true
    assert dash.resources.pool.capacity_gb == 120.0  # still true
    assert dash.errors["stats"].kind == "timeout"


def test_a_missing_system_info_costs_the_pool_but_not_the_table():
    dash = build_dashboard(
        health=_fixture("health"), system_stats=_fixture("system-stats"),
        system_info=None, stats=None, params=None, now=NOW,
    )
    assert dash.resources.pool.capacity_gb is None
    assert dash.resources.pool.percent is None    # no bar against a guess
    assert dash.resources.pool.used_gb is not None  # the numerator still reads
    assert len(dash.models) == 3


def test_no_system_info_means_no_recipes_but_still_no_fabricated_slots():
    """derive_concurrency without the recipes block: enrichment lost,
    correctness kept."""
    dash = build_dashboard(
        health=_fixture("health"), system_stats=None, system_info=None,
        stats=None, params=None, now=NOW,
    )
    assert _row(dash, "Gemma-4-26B").concurrency.display == "8 slots × 64K"


def test_everything_empty_still_renders():
    dash = build_dashboard(None, None, None, None, None, reachable=False, now=NOW)
    assert dash.models == []
    assert dash.reachable is False
    assert dash.resources.pool.capacity_gb is None


def test_a_dead_backend_reads_dead_even_with_no_crash_history():
    """The DB is an enrichment. If it is unavailable, a dead backend must
    still render DEAD rather than inheriting the router's stale 'ready'."""
    dash = build_dashboard(
        health={"all_models_loaded": [
            {"model_name": "x", "status": "ready", "backend_alive": False, "recipe": "llamacpp"},
        ]},
        system_stats=None, system_info=None, stats=None, params=None,
        backend_states={},  # DB read failed
        now=NOW,
    )
    assert dash.models[0].status == "dead"


def test_junk_in_all_models_loaded_does_not_take_down_the_table():
    dash = build_dashboard(
        health={"all_models_loaded": ["nonsense", None, {"model_name": "real"}]},
        system_stats=None, system_info=None, stats=None, params=None, now=NOW,
    )
    assert [r.model_name for r in dash.models] == ["real"]
