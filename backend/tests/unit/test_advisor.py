"""Model Advisor Layer 1 tests (14c §1.1).

This is the layer that must never lie, so it is tested hardest. Every case is a
real model string off the box.
"""
from __future__ import annotations

import pytest

from kblocallm.domain.advisor import (
    FitVerdict,
    QuantClass,
    compute_fit,
    decode_quant,
    parse_params,
    quant_size_impact,
)


# -- quant decoder ---------------------------------------------------------


def test_the_26b_is_decoded_from_its_checkpoint():
    """unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M, the pinned model."""
    q = decode_quant("unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M")
    assert q is not None
    assert q.label == "Q4_K_M"
    assert q.cls is QuantClass.BALANCED
    assert q.k_quant is True
    assert q.bytes_per_weight == 0.56
    assert "sweet spot" in q.explanation


def test_q8_reads_as_near_lossless():
    q = decode_quant("gemma-4-E2B-it-Q8_0")
    assert q.cls is QuantClass.NEAR_LOSSLESS
    assert q.bytes_per_weight == 1.0
    assert "near-lossless" in q.explanation


def test_the_q4_tradeoff_mentions_the_120gb_headroom():
    """The 'you are not forced to q4 on this box' framing is the point of the
    decoder for Kevin."""
    q = decode_quant("model-Q4_K_M")
    assert "120GB" in q.tradeoff and "affordable" in q.tradeoff


def test_the_npu_kernel_quant_is_decoded_distinctly():
    q = decode_quant("something-q4nx")
    assert q.npu_kernel is True
    assert "not a GGUF quant" in q.explanation


def test_xl_variant_is_recognised():
    q = decode_quant("Qwen3.5-4B-UD-Q4_K_XL.gguf")
    assert q.label == "Q4_K_XL"
    assert q.bytes_per_weight == 0.62  # slightly above q4_K_M


def test_a_bare_flm_name_has_no_quant():
    """qwen3.5:2b carries no quant token. That is 'not stated', not a default."""
    assert decode_quant("qwen3.5:2b") is None
    assert decode_quant(None) is None


def test_an_unknown_quant_is_named_not_invented():
    q = decode_quant("model-Q7_K_M")  # valid shape, family we have no figure for
    assert q is not None and q.label == "Q7_K_M"
    assert q.bytes_per_weight is None  # no bytes figure fabricated


# -- param parser ----------------------------------------------------------


def test_moe_params_split_total_and_active():
    """26B-A4B: 26B total in memory, 4B active per token. The distinction
    matters — footprint tracks total, speed tracks active."""
    p = parse_params("Gemma-4-26B-A4B-it-MTP-GGUF")
    assert p.total_b == 26.0 and p.active_b == 4.0 and p.is_moe


def test_dense_model_active_equals_total():
    p = parse_params("Hermes-3-Llama-3.1-8B")
    assert p.total_b == 8.0 and p.active_b == 8.0 and not p.is_moe


def test_sub_billion_params():
    assert parse_params("Aevum-0.6B-Finetuned").total_b == 0.6
    assert parse_params("embed-gemma-300m").total_b == pytest.approx(0.3)


def test_no_param_count_is_none_not_zero():
    assert parse_params("some-model-no-size") is None


# -- fit -------------------------------------------------------------------


def test_the_26b_fits_resident_in_the_pool():
    """17.3 GB against a 120 GB pool with ~56 in use: fits alongside."""
    fit = compute_fit(size_gb=17.3, pool_gb=120.0, used_gb=56.4)
    assert fit.verdict is FitVerdict.FITS_RESIDENT
    assert fit.measured is False   # from disk size, an honest proxy
    assert "120 GB pool" in fit.detail


def test_a_measured_delta_is_preferred_and_labeled():
    fit = compute_fit(size_gb=17.3, pool_gb=120.0, used_gb=56.4, measured_gb=19.1)
    assert fit.footprint_gb == 19.1 and fit.measured is True
    assert "measured resident use" in fit.detail


def test_a_model_that_needs_eviction_says_so():
    fit = compute_fit(size_gb=80.0, pool_gb=120.0, used_gb=56.4)
    assert fit.verdict is FitVerdict.FITS_IF_EVICT
    assert "evict" in fit.detail


def test_a_model_bigger_than_the_pool_will_not_load():
    fit = compute_fit(size_gb=200.0, pool_gb=120.0, used_gb=0.0)
    assert fit.verdict is FitVerdict.TOO_BIG


def test_no_size_no_pool_reads_unknown_never_fabricated():
    """FLM models carry no size. The honest output is 'load it and I can
    measure', not a made-up footprint."""
    fit = compute_fit(size_gb=None, pool_gb=120.0)
    assert fit.verdict is FitVerdict.UNKNOWN
    assert "no catalog size" in fit.detail
    assert compute_fit(size_gb=17.3, pool_gb=None).verdict is FitVerdict.UNKNOWN


# -- size impact -----------------------------------------------------------


def test_size_impact_is_concrete():
    """17.3 GB at q4_K_M (0.56) scaled to q8 (1.0) ~ 30.9 GB."""
    q = decode_quant("Q4_K_M")
    impact = quant_size_impact(17.3, q, target="q8_0")
    assert "17.3 GB" in impact and "30.9 GB" in impact


def test_size_impact_refuses_without_a_known_bpw():
    q = decode_quant("Q7_K_ZZ")  # unknown bpw
    assert quant_size_impact(17.3, q, target="q8_0") is None
