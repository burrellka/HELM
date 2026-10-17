"""`--no-mmap` and friends: removed flags must be blocked before the load."""
from __future__ import annotations

from kblocallm.domain.llamacpp_args import find_deprecated
from kblocallm.domain.load_plan import Level, LoadIntent, plan_load


def test_find_deprecated_pairs_with_replacement():
    assert find_deprecated("--no-mmap") == [("--no-mmap", "--load-mode none")]
    assert find_deprecated("--flash-attn on --no-mmap") == [("--no-mmap", "--load-mode none")]
    assert find_deprecated("--parallel 8") == []
    assert find_deprecated("") == []
    assert find_deprecated(None) == []


def test_find_deprecated_dedups():
    assert find_deprecated("--no-mmap --no-mmap") == [("--no-mmap", "--load-mode none")]


def test_plan_load_blocks_no_mmap():
    plan = plan_load(
        LoadIntent(model_name="m", recipe="llamacpp", slots=2, extra_args="--no-mmap"),
    )
    assert plan.blocked
    err = next(i for i in plan.errors if i.code == "deprecated_arg")
    assert "--load-mode none" in err.message
    assert err.level is Level.ERROR


def test_plan_load_allows_load_mode_none():
    plan = plan_load(
        LoadIntent(model_name="m", recipe="llamacpp", slots=2, extra_args="--load-mode none"),
    )
    assert not any(i.code == "deprecated_arg" for i in plan.issues)
