"""Phase 3 tests: the F9 join, Rule 10 badges, §5.3a search, §5.3 drift.

Driven off the real 162-entry catalog captured from the Lemonade host.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kblocallm.domain.drift import DriftKind, compare_options
from kblocallm.domain.library import (
    LabelJoin,
    build_library,
    search,
    unmatched_loaded_models,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads((FIXTURES / "models.json").read_text())


@pytest.fixture(scope="module")
def health() -> dict:
    return json.loads((FIXTURES / "health.json").read_text())


@pytest.fixture(scope="module")
def entries(catalog, health):
    return build_library(catalog, health)


def _by_id(entries, model_id):
    return next(e for e in entries if e.model_id == model_id)


# -- the F9 join -----------------------------------------------------------


def test_the_live_join_holds_for_every_loaded_model(catalog, health):
    """14a RESULTS §1 confirmed 2/2. Re-confirmed 3/3 here, and this test is
    what notices if the key spaces ever diverge."""
    assert unmatched_loaded_models(catalog, health) == []


def test_loaded_state_lands_on_the_right_catalog_rows(entries):
    assert _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF").loaded is True
    assert _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF").pinned is True
    assert _by_id(entries, "gemma4-it-e2b-FLM").loaded is False


def test_a_failed_join_is_surfaced_not_swallowed():
    """A loaded model with no catalog entry means HELM's badges are blind for
    it. Say which rows we cannot vouch for."""
    unmatched = unmatched_loaded_models(
        {"data": [{"id": "known"}]},
        {"all_models_loaded": [{"model_name": "known"}, {"model_name": "mystery"}]},
    )
    assert unmatched == ["mystery"]


# -- Rule 10: the badge that prevents silent failure -----------------------


def test_flm_models_without_tool_calling_are_flagged_as_silent_strippers(entries):
    """The whole of Rule 10. These four are downloaded RIGHT NOW and would
    silently strip the tools array from any agent wired to them."""
    for model_id in ["llama3.2-1b-FLM", "phi4-mini-it-4b-FLM",
                     "qwen2.5-it-3b-FLM", "lfm2.5-it-1.2b-FLM"]:
        entry = _by_id(entries, model_id)
        assert entry.label_join is LabelJoin.NO_LABELS
        assert entry.strips_tools_silently is True, f"{model_id} must be flagged"


def test_flm_models_with_tool_calling_are_not_flagged(entries):
    for model_id in ["gemma4-it-e2b-FLM", "qwen3.5-2b-FLM"]:
        entry = _by_id(entries, model_id)
        assert "tool-calling" in entry.labels
        assert entry.strips_tools_silently is False


def test_an_embedding_model_is_not_warned_about_tool_stripping(entries):
    """embed-gemma-300m-FLM has no tool-calling label and never will — it is an
    embedding model. The warning is technically true and operationally noise,
    and a warning that fires where it cannot matter is how the warning that
    DOES matter gets ignored."""
    entry = _by_id(entries, "embed-gemma-300m-FLM")
    assert entry.labels == ["embeddings"]
    assert entry.strips_tools_silently is False


def test_a_llamacpp_model_without_the_label_is_not_a_silent_stripper(entries):
    """The strip behaviour is FLM's. Claiming it for llamacpp would be a lie
    in the other direction — and Hermes-3-8B, which is loaded and serving, has
    no tool-calling label."""
    entry = _by_id(entries, "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M")
    assert "tool-calling" not in entry.labels
    assert entry.strips_tools_silently is False


def test_an_unknown_join_never_claims_tools_are_stripped():
    """F9's miss behaviour, in the dangerous direction. If we don't know the
    labels, we must not assert a capability verdict either way."""
    entries = build_library({"data": [{"id": "x", "recipe": "flm"}]})  # no labels key at all
    entry = entries[0]
    assert entry.label_join is LabelJoin.UNKNOWN
    assert entry.strips_tools_silently is False


def test_empty_labels_and_a_failed_join_are_different_states():
    """The core F9 distinction: `labels: []` is an ANSWER; a missing join is
    an absence of one. They must never collapse."""
    known_none = build_library({"data": [{"id": "a", "recipe": "flm", "labels": []}]})[0]
    unknown = build_library({"data": [{"id": "b", "recipe": "flm"}]})[0]
    assert known_none.label_join is LabelJoin.NO_LABELS
    assert unknown.label_join is LabelJoin.UNKNOWN
    assert known_none.strips_tools_silently is True   # a real, actionable finding
    assert unknown.strips_tools_silently is False     # we simply don't know


def test_provenance_labels_are_not_badged_as_capabilities(entries):
    """`custom` and `hot` are curation markers. Badging them next to
    tool-calling dilutes the one badge that matters."""
    entry = _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF")
    assert "hot" in entry.raw_labels          # published
    assert "hot" not in entry.labels          # but not a capability
    assert {"tool-calling", "vision", "mtp"} <= set(entry.labels)


# -- SAVED vs RUNNING drift -----------------------------------------------


def test_the_26b_is_running_unpersisted_sampling_settings(entries):
    """Live drift, today: SAVED '--parallel 8' vs RUNNING '--parallel 8 --temp
    1.0 --top-k 64 --top-p 0.95'. A lemond restart silently drops the sampling
    settings and nobody would notice until the output changed."""
    report = _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF").drift
    args = next(f for f in report.fields if f.field == "llamacpp_args")
    assert args.kind is DriftKind.DRIFT
    assert "--temp 1.0" in args.running and "--temp" not in args.saved


def test_aevum_has_no_saved_configuration_at_all(entries):
    """SAVED {} vs a full RUNNING config. Everything about how it runs is
    ephemeral."""
    report = _by_id(entries, "Aevum-0.6B-Finetuned.Q4_K_M").drift
    assert report.has_unpersisted
    ctx = next(f for f in report.fields if f.field == "ctx_size")
    assert ctx.kind is DriftKind.NOT_SAVED
    assert "restart" in ctx.note


def test_hermes_matches_and_reports_no_drift(entries):
    """The control case. Saved and running agree, so the row must be quiet."""
    report = _by_id(entries, "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M").drift
    assert report.has_drift is False


def test_auto_detected_backend_does_not_cry_wolf(entries):
    """llamacpp_backend is absent from SAVED on every model on the box because
    Lemonade auto-detects it. Flagging that as drift would light up every row,
    and an indicator that is always on is one nobody reads."""
    for model_id in ["Gemma-4-26B-A4B-it-MTP-GGUF", "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"]:
        report = _by_id(entries, model_id).drift
        backend = next(f for f in report.fields if f.field == "llamacpp_backend")
        assert backend.kind is DriftKind.NOT_SAVED
        assert backend.benign is True
        assert backend not in report.notable
    assert _by_id(entries, "Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M").drift.has_unpersisted is False


def test_slot_drift_is_derived_through_the_shared_parser():
    """8 saved vs 4 running is the drift that matters most: it changes the
    per-slot context, i.e. the number Rule 1 exists to protect."""
    report = compare_options(
        {"llamacpp_args": "--parallel 8"}, {"llamacpp_args": "-np 4"},
    )
    slots = next(f for f in report.fields if f.field == "slots")
    assert (slots.saved, slots.running) == (8, 4)
    assert slots.kind is DriftKind.DRIFT


def test_unparseable_running_args_do_not_fake_a_slot_match():
    report = compare_options({"llamacpp_args": "--parallel 8"}, {"llamacpp_args": "--parallel abc"})
    slots = next(f for f in report.fields if f.field == "slots")
    assert slots.running is None


# -- §5.3 settings, derived server-side ------------------------------------


def test_the_26b_settings_are_derived_in_the_operators_units(entries):
    """What the settings editor prefills from. The operator thinks "8 slots of
    64K"; Lemonade stores 524288. The division happens HERE, next to its test
    table, not in a React component."""
    running = _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF").running_settings
    assert running.slots == 8
    assert running.ctx_per_slot == 65536
    assert running.ctx_size == 524288       # the raw total, for reference only
    assert running.backend == "rocm"
    # Sampling params are lifted into their own labeled fields, so the editor
    # shows them as inputs rather than buried in a raw text box.
    assert running.temperature == 1.0
    assert running.top_k == 64
    assert running.top_p == 0.95
    assert running.min_p is None            # absent -> unset, not zero
    assert running.repeat_penalty is None
    # --parallel AND the sampling flags are stripped: each field owns its flag,
    # and the planner refuses a value with two sources. Nothing left over here.
    assert running.extra_args == ""


def test_saved_and_running_settings_differ_where_the_drift_says_they_do(entries):
    entry = _by_id(entries, "Gemma-4-26B-A4B-it-MTP-GGUF")
    assert entry.saved_settings.temperature is None         # nothing persisted
    assert entry.running_settings.temperature == 1.0        # but running with sampling
    assert entry.saved_settings.slots == entry.running_settings.slots == 8


def test_aevum_defaults_to_one_slot_and_the_full_context(entries):
    running = _by_id(entries, "Aevum-0.6B-Finetuned.Q4_K_M").running_settings
    assert running.slots == 1                # no --parallel: llama.cpp's default
    assert running.ctx_per_slot == 40960     # 40960 / 1


def test_an_unreadable_slot_count_refuses_to_derive_a_per_slot_context():
    """The 8x lie, guarded at the settings layer too: falling back to 1 slot
    would render the TOTAL as the per-slot value."""
    entries = build_library({"data": [{
        "id": "x", "recipe": "llamacpp",
        "recipe_options": {"ctx_size": 524288, "llamacpp_args": "--parallel abc"},
    }]})
    settings = entries[0].saved_settings
    assert settings.slots is None
    assert settings.ctx_per_slot is None
    assert settings.ctx_size == 524288


def test_flm_settings_carry_context_but_no_slots():
    entries = build_library({"data": [{
        "id": "x", "recipe": "flm", "recipe_options": {"ctx_size": 131072},
    }]})
    settings = entries[0].saved_settings
    assert settings.slots is None
    assert settings.ctx_per_slot == 131072   # serial: the context is the context


# -- §5.3a search ----------------------------------------------------------


def test_the_uncensored_model_can_be_tagged_and_found(catalog, health):
    """Kevin's real risk: the lorablated 8B sits in the picker next to the
    standard one. Tag it once, find it forever."""
    meta = {"hermes-3-llama-3.1-8b-lorablated.Q4_K_M": {"tags": ["uncensored-do-not-deploy"]}}
    entries = build_library(catalog, health, meta)
    hits = search(entries, tags={"uncensored-do-not-deploy"})
    assert [e.model_id for e in hits] == ["hermes-3-llama-3.1-8b-lorablated.Q4_K_M"]


def test_the_spec_example_query(catalog, health):
    """'favorited, downloaded, tool-calling models tagged hermes-preferred'."""
    meta = {"Gemma-4-26B-A4B-it-MTP-GGUF": {"favorite": True, "tags": ["hermes-preferred"]}}
    entries = build_library(catalog, health, meta)
    hits = search(entries, downloaded=True, favorite=True,
                  labels={"tool-calling"}, tags={"hermes-preferred"})
    assert [e.model_id for e in hits] == ["Gemma-4-26B-A4B-it-MTP-GGUF"]


def test_all_npu_models_with_the_embeddings_label(catalog, health):
    entries = build_library(catalog, health)
    hits = search(entries, recipes={"flm"}, labels={"embeddings"})
    assert [e.model_id for e in hits] == ["embed-gemma-300m-FLM"]


def test_archived_models_are_hidden_unless_asked_for(catalog, health):
    """The LocalAI 'disabled' idea: shelve a model without deleting 20GB."""
    meta = {"gemma4-it-e2b-FLM": {"archived": True}}
    entries = build_library(catalog, health, meta)
    assert not any(e.model_id == "gemma4-it-e2b-FLM" for e in search(entries))
    assert any(e.model_id == "gemma4-it-e2b-FLM"
               for e in search(entries, include_archived=True))


def test_free_text_searches_description_and_checkpoint(catalog, health):
    meta = {"qwen3.5-2b-FLM": {"description": "Karl's travel brain"}}
    entries = build_library(catalog, health, meta)
    assert [e.model_id for e in search(entries, text="karl's travel")] == ["qwen3.5-2b-FLM"]
    assert any(e.model_id == "Gemma-4-26B-A4B-it-MTP-GGUF"
               for e in search(entries, text="unsloth/gemma-4"))


def test_multiple_labels_and_together(catalog, health):
    entries = build_library(catalog, health)
    both = search(entries, labels={"tool-calling", "vision"}, downloaded=True)
    assert all({"tool-calling", "vision"} <= set(e.labels) for e in both)
    assert any(e.model_id == "Gemma-4-26B-A4B-it-MTP-GGUF" for e in both)


def test_favourites_sort_to_the_top(catalog, health):
    meta = {"qwen3.5-2b-FLM": {"favorite": True}}
    entries = build_library(catalog, health, meta)
    assert search(entries)[0].model_id == "qwen3.5-2b-FLM"


def test_the_whole_catalog_survives_a_plain_search(entries):
    """162 entries, 25 downloaded. Sanity that nothing silently drops rows."""
    assert len(search(entries)) == 162
    assert len(search(entries, downloaded=True)) == 25


def test_junk_catalog_entries_are_skipped_not_fatal():
    assert [e.model_id for e in build_library({"data": ["junk", None, {"id": "ok"}]})] == ["ok"]


def test_no_catalog_is_an_empty_library_not_a_crash():
    assert build_library(None) == []
    assert build_library({}) == []
