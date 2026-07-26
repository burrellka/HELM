"""Phase 2 planning tests.

The NPU-exclusivity and ctx-multiplication cases are the ones that matter: both
are silent, both are destructive, and both happen at the moment the operator is
least able to catch them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kblocallm.domain.load_plan import LoadIntent, Level, plan_load, plan_unload

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def health() -> dict:
    return json.loads((FIXTURES / "health.json").read_text())


def _codes(plan, level=None):
    return {i.code for i in plan.issues if level is None or i.level is level}


# -- the multiplication ----------------------------------------------------


def test_helm_does_the_ctx_multiplication_not_the_operator():
    """The operator asks for 8 slots of 64K. Lemonade wants ctx_size 524288.
    That translation is HELM's whole job here (§5.3)."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=8, ctx_per_slot=65536))
    assert plan.computed_ctx_size == 524288
    assert plan.body["ctx_size"] == 524288
    assert plan.body["llamacpp_args"] == "--parallel 8"
    assert not plan.blocked


def test_the_computed_total_is_explained_not_just_sent():
    """Showing the total as a consequence is what stops the next person
    believing ctx_size is per-slot."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=8, ctx_per_slot=65536))
    note = next(i for i in plan.issues if i.code == "ctx_computed")
    assert "524,288" in note.message and "65,536 per slot" in note.message


def test_single_slot_needs_no_multiplication():
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=1, ctx_per_slot=40960))
    assert plan.computed_ctx_size == 40960
    assert "ctx_computed" not in _codes(plan)  # nothing to explain


def test_ctx_over_the_models_ceiling_is_blocked():
    plan = plan_load(
        LoadIntent(model_name="m", recipe="llamacpp", slots=2, ctx_per_slot=999_999),
        catalog={"max_context_window": 131072},
    )
    assert plan.blocked and "ctx_over_ceiling" in _codes(plan, Level.ERROR)


def test_no_ceiling_known_means_no_false_block():
    """Missing catalog data costs a cross-check, never correctness."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=2, ctx_per_slot=999_999))
    assert not plan.blocked


# -- NPU exclusivity -------------------------------------------------------


def _npu_health(*names):
    return {"all_models_loaded": [
        {"model_name": n, "recipe": "flm", "device": "npu"} for n in names
    ]}


def test_loading_an_npu_model_names_the_victim():
    """Rule 7. The router WILL unload the incumbent, silently. Naming it is
    the difference between an informed decision and an agent losing its brain
    mid-conversation."""
    plan = plan_load(
        LoadIntent(model_name="new-npu", recipe="flm"),
        health=_npu_health("qwen3.5-2b-FLM"),
    )
    assert plan.evicts == ["qwen3.5-2b-FLM"]
    warning = next(i for i in plan.issues if i.code == "npu_exclusivity")
    assert "will UNLOAD qwen3.5-2b-FLM" in warning.message
    # A warning, not an error: it is a legitimate action with a consequence.
    assert not plan.blocked


def test_loading_a_gpu_model_evicts_no_npu_model(health):
    """The exclusivity rule is NPU-only. Warning about it on a llamacpp load
    would train the operator to click through the warning that matters."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp"), health=health)
    assert plan.evicts == []
    assert "npu_exclusivity" not in _codes(plan)


def test_reloading_the_same_npu_model_does_not_warn_about_itself():
    plan = plan_load(
        LoadIntent(model_name="qwen3.5-2b-FLM", recipe="flm"),
        health=_npu_health("qwen3.5-2b-FLM"),
    )
    assert plan.evicts == []


def test_ryzenai_llm_is_also_npu():
    """system-info gives ryzenai-llm slot_policy exclusive_npu. A denylist of
    just 'flm' would miss it."""
    plan = plan_load(
        LoadIntent(model_name="x", recipe="ryzenai-llm"),
        health=_npu_health("qwen3.5-2b-FLM"),
    )
    assert plan.evicts == ["qwen3.5-2b-FLM"]


# -- reserved args ---------------------------------------------------------


@pytest.mark.parametrize("args", ["--ctx-size 4096", "-c 4096", "--ctx-size=4096", "--jinja"])
def test_reserved_args_are_blocked_in_every_spelling(args):
    """Same parser as the slot math (F4). The server hard-errors on these, so
    catching them here saves a round trip AND explains why."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", extra_args=args))
    assert plan.blocked and "reserved_args" in _codes(plan, Level.ERROR)


def test_the_reserved_error_says_what_to_do_instead():
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", extra_args="-c 4096"))
    message = next(i for i in plan.errors if i.code == "reserved_args").message
    assert "context-per-slot" in message


def test_harmless_args_pass_through():
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=2, extra_args="--temp 1.0 --top-k 64",
    ))
    assert not plan.blocked
    assert plan.body["llamacpp_args"] == "--parallel 2 --temp 1.0 --top-k 64"


def test_parallel_in_both_places_is_refused_rather_than_silently_resolved():
    """Two sources for one value, and Lemonade's arg order decides which wins.
    Whichever HELM picked, the displayed slot count could disagree with the
    running one — a silent divergence."""
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=8, extra_args="--parallel 2",
    ))
    assert plan.blocked and "duplicate_parallel" in _codes(plan, Level.ERROR)


# -- sampling fields --------------------------------------------------------


def test_sampling_fields_compose_into_llamacpp_args():
    """The labeled fields ARE the source; HELM spells the flags. The operator
    never types `--temp` — they fill a number box and HELM composes it back."""
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=2,
        temperature=0.7, top_k=40, top_p=0.9, min_p=0.05, repeat_penalty=1.1,
    ))
    assert not plan.blocked
    assert plan.body["llamacpp_args"] == (
        "--parallel 2 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1"
    )


def test_a_whole_number_sampling_value_has_no_trailing_dot_zero():
    """`repeat_penalty=1.0` must emit `--repeat-penalty 1`, so the parser reads
    it back identically and the drift check does not flag a phantom 1 vs 1.0."""
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=1, temperature=1.0, repeat_penalty=1.0,
    ))
    args = plan.body["llamacpp_args"]
    assert "--temp 1 " in args + " " and "--repeat-penalty 1" in args
    assert "1.0" not in args


def test_unset_sampling_fields_emit_nothing():
    """Empty field = leave unset = the flag is absent. A blank temp box must not
    become `--temp 0` or `--temp None`."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=2))
    assert plan.body["llamacpp_args"] == "--parallel 2"


def test_a_sampling_param_in_both_field_and_raw_args_is_refused():
    """Same rule as --parallel: one value, one source. temp in its field AND in
    the escape hatch is a silent disagreement the moment their order matters."""
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=1,
        temperature=0.7, extra_args="--temp 0.2",
    ))
    assert plan.blocked and "duplicate_sampling" in _codes(plan, Level.ERROR)
    message = next(i for i in plan.errors if i.code == "duplicate_sampling").message
    assert "--temp" in message


def test_a_sampling_param_only_in_raw_args_still_passes():
    """The escape hatch is still open: an operator who types --temp themselves,
    and leaves the field blank, is not fighting a second source. Allow it."""
    plan = plan_load(LoadIntent(
        model_name="m", recipe="llamacpp", slots=1, extra_args="--temp 0.2",
    ))
    assert not plan.blocked
    assert plan.body["llamacpp_args"] == "--parallel 1 --temp 0.2"


# -- slots are llamacpp-only -----------------------------------------------


def test_slots_on_an_npu_model_are_blocked():
    """§5.3: 'recipe flm -> hide slot controls entirely. Showing a slot field
    would be a lie.' If one arrives anyway, refuse it."""
    plan = plan_load(LoadIntent(model_name="m", recipe="flm", slots=4))
    assert plan.blocked and "slots_not_supported" in _codes(plan, Level.ERROR)
    assert "one request at a time" in next(
        i for i in plan.errors if i.code == "slots_not_supported").message


def test_slots_on_whispercpp_are_blocked_too():
    plan = plan_load(LoadIntent(model_name="m", recipe="whispercpp", slots=4))
    assert plan.blocked and "slots_not_supported" in _codes(plan, Level.ERROR)


def test_an_npu_load_still_sets_ctx_without_slots():
    plan = plan_load(LoadIntent(model_name="m", recipe="flm", ctx_per_slot=131072))
    assert not plan.blocked
    assert plan.body["ctx_size"] == 131072
    assert "llamacpp_args" not in plan.body


@pytest.mark.parametrize("slots", [0, -1])
def test_nonsense_slot_counts_are_blocked(slots):
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", slots=slots))
    assert plan.blocked and "bad_slots" in _codes(plan, Level.ERROR)


# -- memory honesty --------------------------------------------------------


def test_we_do_not_pretend_to_estimate_memory(health):
    """§5.3 asks for a live memory estimate. Lemonade exposes no per-model
    memory, so any figure would be invented. Say so rather than guess."""
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp"), health=health)
    note = next(i for i in plan.issues if i.code == "no_memory_estimate")
    assert "cannot estimate" in note.message
    assert note.level is Level.INFO


# -- body shape ------------------------------------------------------------


def test_pinned_is_top_level():
    plan = plan_load(LoadIntent(model_name="m", recipe="llamacpp", pinned=True))
    assert plan.body["pinned"] is True


def test_save_options_is_explicit():
    """--save-options is the difference between 'runs with' and 'persists as'.
    §5.3's SAVED-vs-RUNNING drift exists because they are not the same."""
    assert "save_options" not in plan_load(LoadIntent(model_name="m")).body
    assert plan_load(LoadIntent(model_name="m", save_options=True)).body["save_options"] is True


def test_a_bare_load_sends_a_minimal_body():
    """No options means 'let Lemonade auto-size it'. Do not send empty
    structures that read as deliberate settings."""
    plan = plan_load(LoadIntent(model_name="m"))
    assert plan.body == {"model_name": "m"}


def test_no_model_name_is_blocked():
    assert plan_load(LoadIntent(model_name="")).blocked


# -- unload ----------------------------------------------------------------


def test_unloading_a_pinned_model_warns_that_pin_will_not_bring_it_back(health):
    """Rule 5: pinned prevents eviction; it does NOT reload. Unloading the
    pinned 26B leaves it unloaded, and six agents notice."""
    plan = plan_unload("Gemma-4-26B-A4B-it-MTP-GGUF", health)
    codes = _codes(plan)
    assert "unloading_pinned" in codes and "in_flight" in codes
    assert "will stay unloaded" in next(
        i for i in plan.issues if i.code == "unloading_pinned").message


def test_unloading_an_unpinned_model_warns_only_about_in_flight(health):
    plan = plan_unload("Aevum-0.6B-Finetuned.Q4_K_M", health)
    assert "unloading_pinned" not in _codes(plan)
    assert "in_flight" in _codes(plan)


def test_unloading_something_not_loaded_says_so(health):
    plan = plan_unload("not-loaded", health)
    assert "not_loaded" in _codes(plan)


def test_unload_is_never_blocked_only_warned(health):
    """HELM's job is to inform, not to refuse. Unloading the 26B is a real
    thing an operator legitimately needs to do."""
    assert not plan_unload("Gemma-4-26B-A4B-it-MTP-GGUF", health).blocked
