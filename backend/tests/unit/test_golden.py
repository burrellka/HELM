"""Golden config: build extraction + drift comparison. Pure, table-tested."""
from __future__ import annotations

from kblocallm.domain.golden import (
    GoldenConfig,
    GoldenState,
    capture_from_record,
    compare_golden,
    extract_backend_build,
)

GOLDEN = GoldenConfig(
    model_id="gemma",
    ctx_size=770048,
    llamacpp_backend="rocm",
    backend_build="llama-b11394",
    llamacpp_args="--parallel 8 --reasoning off --temp 1.0 --top-k 64 --top-p 0.95",
)


def _record(args, *, ctx=770048, backend="rocm", build="llama-b11394"):
    lc = [f"/var/cache/lemonade/bin/llamacpp/rocm-stable/{build}/llama-server", "-m", "f.gguf"]
    return {
        "recipe_options": {"ctx_size": ctx, "llamacpp_backend": backend, "llamacpp_args": args},
        "launch_command": lc,
    }


# --- extract_backend_build ------------------------------------------------

def test_extract_build_from_argv_list():
    assert extract_backend_build(
        ["/x/rocm-stable/llama-b11394/llama-server", "-m", "f"]) == "llama-b11394"


def test_extract_build_from_string():
    assert extract_backend_build("… /llama-b9001/llama-server …") == "llama-b9001"


def test_extract_build_absent_or_bad():
    assert extract_backend_build(["/usr/bin/llama-server"]) is None
    assert extract_backend_build(None) is None
    assert extract_backend_build(42) is None


# --- compare_golden: the core --------------------------------------------

def test_match_when_running_equals_golden_plus_injected_flags():
    # Running has Lemonade's INJECTED companions (--spec-type/--mmproj) on top of
    # the golden args. Those are not tracked, so this is a MATCH, not drift.
    rec = _record("--parallel 8 --reasoning off --temp 1.0 --top-k 64 --top-p 0.95 "
                  "--spec-type draft-mtp --slot-prompt-similarity 0.9")
    assert compare_golden(GOLDEN, rec).state is GoldenState.MATCH


def test_dropped_reasoning_is_critical_drift():
    # The headline case: a reload drops --reasoning off → thinking comes back on.
    rec = _record("--parallel 8 --temp 1.0 --top-k 64 --top-p 0.95")
    drift = compare_golden(GOLDEN, rec)
    assert drift.state is GoldenState.DRIFT
    assert drift.has_critical
    item = next(i for i in drift.items if i.key == "reasoning")
    assert item.golden == "off" and item.running is None and item.critical


def test_reasoning_flipped_on_is_critical_drift():
    rec = _record("--parallel 8 --reasoning on --temp 1.0 --top-k 64 --top-p 0.95")
    drift = compare_golden(GOLDEN, rec)
    assert drift.has_critical
    assert any(i.key == "reasoning" and i.running == "on" for i in drift.items)


def test_changed_sampling_is_non_critical_drift():
    rec = _record("--parallel 8 --reasoning off --temp 0.6 --top-k 64 --top-p 0.95")
    drift = compare_golden(GOLDEN, rec)
    assert drift.state is GoldenState.DRIFT
    assert not drift.has_critical
    assert any(i.key == "temp" and i.golden == "1.0" and i.running == "0.6" for i in drift.items)


def test_slot_count_drift():
    rec = _record("--parallel 1 --reasoning off --temp 1.0 --top-k 64 --top-p 0.95")
    drift = compare_golden(GOLDEN, rec)
    assert any(i.key == "parallel" and i.golden == "8" and i.running == "1" for i in drift.items)


def test_backend_build_change_is_flagged_not_critical():
    rec = _record("--parallel 8 --reasoning off --temp 1.0 --top-k 64 --top-p 0.95",
                  build="llama-b12000")
    drift = compare_golden(GOLDEN, rec)
    assert drift.state is GoldenState.DRIFT
    assert not drift.has_critical
    assert any(i.key == "backend_build" and i.golden == "llama-b11394"
               and i.running == "llama-b12000" for i in drift.items)


def test_ctx_and_backend_drift():
    rec = _record("--parallel 8 --reasoning off --temp 1.0 --top-k 64 --top-p 0.95",
                  ctx=131072, backend="vulkan")
    keys = {i.key for i in compare_golden(GOLDEN, rec).items}
    assert "ctx_size" in keys and "backend" in keys


def test_unknown_when_running_unreadable():
    # No recipe_options AND no parseable build -> cannot verify, say so.
    assert compare_golden(GOLDEN, {"foo": "bar"}).state is GoldenState.UNKNOWN


def test_golden_without_a_flag_does_not_hold_the_line_on_it():
    # A golden that never pinned --min-p must not flag a running --min-p.
    g = GoldenConfig(model_id="m", llamacpp_args="--reasoning off")
    rec = _record("--reasoning off --min-p 0.1")
    assert compare_golden(g, rec).state is GoldenState.MATCH


# --- capture_from_record --------------------------------------------------

def test_capture_snapshots_current_config():
    rec = _record("--parallel 8 --reasoning off", ctx=770048, backend="rocm")
    g = capture_from_record("gemma", rec, note="post-upgrade blessed")
    assert g.ctx_size == 770048
    assert g.llamacpp_backend == "rocm"
    assert g.backend_build == "llama-b11394"
    assert "--reasoning off" in g.llamacpp_args
    assert g.note == "post-upgrade blessed"
    # And it round-trips to a MATCH against the record it was captured from.
    assert compare_golden(g, rec).state is GoldenState.MATCH
