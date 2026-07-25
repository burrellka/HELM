"""Running-context drift (14c §1.12), anchored on the live box's real records.

The three llamacpp models come verbatim from `tests/fixtures/health.json`. They
carry three DISTINCT slot counts (26B=8, Hermes=2, Aevum=1-default), which is
exactly what lets HELM attribute Lemonade's model-blind `/slots` to one model by
its slot count. When two share a count, attribution is impossible and the answer
must be `unknown` — these tests hold that line, because a wrong running-context
number stated confidently is the precise failure §6 exists to prevent.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kblocallm.domain.ctx_drift import DriftState, check_ctx_drift

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def loaded() -> dict[str, dict]:
    payload = json.loads((FIXTURES / "health.json").read_text())
    return {m["model_name"]: m for m in payload["all_models_loaded"]}


def slots(n: int, n_ctx: int) -> list[dict]:
    return [{"id": i, "n_ctx": n_ctx, "is_processing": False} for i in range(n)]


# -- the happy path: /slots attributable to exactly one model --------------


def test_26b_match_when_running_ctx_equals_computed(loaded):
    """26B computes 524288/8 = 65536. /slots reports 8 slots of 65536, and the
    26B is the only loaded model with 8 slots, so it is attributable -> MATCH."""
    all_loaded = list(loaded.values())
    r = check_ctx_drift(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], all_loaded, slots(8, 65536))
    assert r.state is DriftState.MATCH
    assert r.computed_per_slot == 65536
    assert r.running_n_ctx == 65536


def test_26b_drift_when_llamacpp_capped_context(loaded):
    """The real prize: HELM computed 65536, but llama.cpp is actually running
    40960 per slot (capped under memory pressure). That is DRIFT, and the
    running value is the truth we surface."""
    all_loaded = list(loaded.values())
    r = check_ctx_drift(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], all_loaded, slots(8, 40960))
    assert r.state is DriftState.DRIFT
    assert r.computed_per_slot == 65536
    assert r.running_n_ctx == 40960


def test_hermes_attributed_by_its_own_slot_count(loaded):
    """Hermes has --parallel 2, 131072/2 = 65536. /slots reporting 2 slots of
    65536 is unambiguously Hermes' (nothing else loaded has 2 slots)."""
    all_loaded = list(loaded.values())
    r = check_ctx_drift(loaded["Hermes-3-Llama-3.1-8B-GGUF-Q4_K_M"], all_loaded, slots(2, 65536))
    assert r.state is DriftState.MATCH
    assert r.computed_per_slot == 65536


# -- the /slots-is-not-per-model reality: unknown, never no-drift ----------


def test_unknown_when_slots_belong_to_a_different_model(loaded):
    """Kevin's actual box: /slots resolves to the 26B (8 slots), but we are
    asking about Aevum (1 slot). The counts do not match, so these slots are
    NOT Aevum's. Drift must be UNKNOWN, never 'no drift' from a number that
    isn't even this model's."""
    all_loaded = list(loaded.values())
    r = check_ctx_drift(loaded["Aevum-0.6B-Finetuned.Q4_K_M"], all_loaded, slots(8, 65536))
    assert r.state is DriftState.UNKNOWN
    assert r.running_n_ctx is None
    assert r.computed_per_slot == 40960  # we still show what HELM computed


def test_unknown_when_two_models_share_the_slot_count(loaded):
    """If two loaded models both have 8 slots, an 8-slot /slots payload cannot
    be attributed to either — /slots carries no model label. UNKNOWN."""
    twin = copy.deepcopy(loaded["Aevum-0.6B-Finetuned.Q4_K_M"])
    twin["model_name"] = "Twin-8slot"
    twin["recipe_options"] = {"ctx_size": 524288, "llamacpp_args": "--parallel 8"}
    all_loaded = [loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], twin]
    r = check_ctx_drift(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], all_loaded, slots(8, 65536))
    assert r.state is DriftState.UNKNOWN
    assert r.running_n_ctx is None


# -- the other unknown gates -----------------------------------------------


def test_unknown_when_not_loaded(loaded):
    unloaded = copy.deepcopy(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"])
    unloaded["loaded"] = False
    r = check_ctx_drift(unloaded, [], slots(8, 65536))
    assert r.state is DriftState.UNKNOWN


def test_unknown_when_recipe_has_no_slots():
    flm = {"recipe": "flm", "loaded": True, "recipe_options": {"ctx_size": 40960}}
    r = check_ctx_drift(flm, [], slots(1, 40960))
    assert r.state is DriftState.UNKNOWN


def test_unknown_when_slots_payload_missing(loaded):
    all_loaded = list(loaded.values())
    r = check_ctx_drift(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], all_loaded, None)
    assert r.state is DriftState.UNKNOWN


def test_unknown_when_parallel_is_malformed(loaded):
    bad = copy.deepcopy(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"])
    bad["recipe_options"]["llamacpp_args"] = "--parallel abc"
    r = check_ctx_drift(bad, [bad], slots(8, 65536))
    # A malformed --parallel never becomes a slot count, so nothing to compare.
    assert r.state is DriftState.UNKNOWN


def test_unknown_when_slots_disagree_on_n_ctx(loaded):
    """A /slots payload whose slots report different n_ctx is not something to
    average into a tidy number — it is unreadable, so: unknown."""
    all_loaded = list(loaded.values())
    mixed = [
        {"id": 0, "n_ctx": 65536, "is_processing": False},
        {"id": 1, "n_ctx": 32768, "is_processing": False},
    ] + slots(6, 65536)
    r = check_ctx_drift(loaded["Gemma-4-26B-A4B-it-MTP-GGUF"], all_loaded, mixed)
    assert r.state is DriftState.UNKNOWN
