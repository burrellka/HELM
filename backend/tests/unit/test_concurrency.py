"""Rule 1 + F3 tests, anchored on the live box's real records.

The three llamacpp models here are copied verbatim out of
`tests/fixtures/health.json`, captured from the Lemonade host on 2026-07-16. If
these go red, the Concurrency column is lying about the models six agents and
three humans are actually using.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kblocallm.domain.concurrency import ConcurrencyKind, derive_concurrency, format_ctx

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def recipes() -> dict:
    return json.loads((FIXTURES / "system-info.json").read_text())["recipes"]


@pytest.fixture(scope="module")
def loaded() -> dict[str, dict]:
    payload = json.loads((FIXTURES / "health.json").read_text())
    return {m["model_name"]: m for m in payload["all_models_loaded"]}


# -- the live models -------------------------------------------------------


def test_the_26b_reads_8_slots_of_64k_not_8_of_512k(loaded, recipes):
    """THE Rule 1 test. ctx_size 524288 with --parallel 8 is 8 x 65,536.
    Reporting 512K per slot is the failure the whole spec is written against.
    """
    result = derive_concurrency(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], recipes)
    assert result.kind is ConcurrencyKind.SLOTS
    assert result.slots == 8
    assert result.per_slot_ctx == 65536
    assert result.display == "8 slots × 64K"
    assert "524,288" in (result.note or ""), "say the raw total is a total"


def test_hermes_8b_reads_2_slots_of_64k(loaded, recipes):
    result = derive_concurrency(loaded["Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"], recipes)
    assert (result.slots, result.per_slot_ctx) == (2, 65536)
    assert result.display == "2 slots × 64K"


def test_aevum_has_no_parallel_flag_and_says_default(loaded, recipes):
    """Live record with sampling args and no --parallel. One slot is the
    right answer, and the display must own that it's llama.cpp's default
    rather than something the operator chose."""
    result = derive_concurrency(loaded["Aevum-0.6B-Finetuned.Q4_K_M"], recipes)
    assert result.kind is ConcurrencyKind.SLOTS
    assert (result.slots, result.per_slot_ctx) == (1, 40960)
    assert result.slots_defaulted is True
    assert result.display == "1 slot (default) × 40K"


def test_raw_ctx_size_never_reaches_the_display(loaded, recipes):
    """Rule 1's negative form: the raw total is misleading, so it must not be
    the number on screen."""
    for model in loaded.values():
        display = derive_concurrency(model, recipes).display
        assert "524288" not in display and "512K" not in display


# -- F3: the allowlist -----------------------------------------------------


def _model(recipe: str, **options):
    return {"recipe": recipe, "recipe_options": options}


def test_flm_is_serial_never_a_slot_count(recipes):
    """Rule 2. ctx_size is present and large; a denylist parser would have
    happily divided it into slots the NPU does not have."""
    result = derive_concurrency(_model("flm", ctx_size=131072), recipes)
    assert result.kind is ConcurrencyKind.SERIAL
    assert result.slots is None
    assert result.display == "SERIAL @ 128K"
    assert result.slot_policy == "coexist_by_type"  # verbatim from the box


@pytest.mark.parametrize("recipe", ["whispercpp", "sd-cpp", "kokoro", "moonshine"])
def test_installable_recipes_get_no_fabricated_slots(recipe, recipes):
    """F3's actual point. §5B.4 pushes installing these, and a denylist
    (`if flm ... else slots`) fabricates a slot count for every one of them.
    """
    result = derive_concurrency(_model(recipe, ctx_size=4096), recipes)
    assert result.kind is ConcurrencyKind.NOT_APPLICABLE
    assert result.slots is None
    assert result.display == "n/a"


def test_vllm_has_slots_but_not_llamacpp_ones(recipes):
    """The subtle one. vllm has parallel semantics and uses_ctx_size, but they
    are NOT spelled in llamacpp_args — so deriving from our parser would be a
    fabrication. The allowlist keeps us out of it."""
    result = derive_concurrency(_model("vllm", ctx_size=8192), recipes)
    assert result.kind is ConcurrencyKind.NOT_APPLICABLE


def test_an_unknown_future_recipe_does_not_fabricate_slots():
    """No system-info metadata, a recipe nobody has thought of yet. Still
    must not produce a number."""
    result = derive_concurrency(_model("some-new-backend", ctx_size=4096), {})
    assert result.kind is ConcurrencyKind.NOT_APPLICABLE
    assert result.slots is None


def test_missing_recipe_reads_unknown():
    result = derive_concurrency({"recipe": "", "recipe_options": {}}, {})
    assert result.kind is ConcurrencyKind.UNKNOWN
    assert result.display == "unknown"


def test_recipes_that_do_not_use_ctx_size_show_no_context(recipes):
    """whispercpp reports uses_ctx_size: false on the live box. A context
    number there would be meaningless."""
    result = derive_concurrency(_model("whispercpp", ctx_size=4096), recipes)
    assert result.per_slot_ctx is None


# -- degradation -----------------------------------------------------------


def test_malformed_args_degrade_to_unknown_never_to_a_number(recipes):
    """F4 meeting Rule 1. A bad parse must not fall back to 1 slot, because
    that renders ctx_size itself as the per-slot value: 8x the truth."""
    result = derive_concurrency(
        _model("llamacpp", ctx_size=524288, llamacpp_args="--parallel abc"), recipes,
    )
    assert result.kind is ConcurrencyKind.UNKNOWN
    assert result.display == "unknown"
    assert result.per_slot_ctx is None
    assert "abc" in (result.note or "")


def test_missing_ctx_size_keeps_the_slot_count_and_admits_the_rest(recipes):
    """Partial knowledge is fine as long as the gap is visible. We know the
    slot count; we don't know the context; say exactly that."""
    result = derive_concurrency(
        _model("llamacpp", llamacpp_args="--parallel 4"), recipes,
    )
    assert result.slots == 4
    assert result.per_slot_ctx is None
    assert result.display == "4 slots × unknown ctx"


@pytest.mark.parametrize("bad", [0, -1, None, "big", True])
def test_junk_ctx_size_does_not_crash_the_row(bad, recipes):
    result = derive_concurrency(
        _model("llamacpp", ctx_size=bad, llamacpp_args="--parallel 8"), recipes,
    )
    assert result.slots == 8 and result.per_slot_ctx is None


def test_uneven_division_keeps_its_digits(recipes):
    """1000 / 3 isn't a round K. Show the real number rather than rounding it
    into a tidier lie."""
    result = derive_concurrency(
        _model("llamacpp", ctx_size=1000, llamacpp_args="--parallel 3"), recipes,
    )
    assert result.per_slot_ctx == 333
    assert result.display == "3 slots × 333"


@pytest.mark.parametrize("ctx,expected", [(65536, "64K"), (40960, "40K"), (1500, "1,500")])
def test_format_ctx(ctx, expected):
    assert format_ctx(ctx) == expected
