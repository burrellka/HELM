"""Per-model /metrics parser (14c §1.12), anchored on the box's real capture.

`tests/fixtures/metrics.txt` is Lemonade's live `/metrics` from the host. The
LFM row in it reports tokens_per_second = 1000000 — the fabricated-rate artifact
— so this suite pins that the parser refuses it exactly as the dashboard does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kblocallm.domain.metrics import parse_model_metrics

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return parse_model_metrics((FIXTURES / "metrics.txt").read_text())


def test_one_entry_per_model(metrics):
    assert {"Gemma-4-26B-A4B-it-MTP-GGUF", "LFM2.5-1.2B-Instruct-GGUF",
            "Aevum-0.6B-Finetuned.Q4_K_M"} <= set(metrics)


def test_26b_real_rate_and_counters(metrics):
    m = metrics["Gemma-4-26B-A4B-it-MTP-GGUF"]
    assert m.tokens_per_second == pytest.approx(60.0, abs=0.1)
    assert m.rate_implausible is False
    assert m.requests_total == 35
    assert m.output_tokens_total == 21594
    assert m.checkpoint == "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M"
    assert m.time_to_first_token == pytest.approx(0.0268, abs=0.001)


def test_lfm_fabricated_rate_is_refused(metrics):
    """THE guard: 1,000,000 tok/s is not a rate. rate_implausible flips and the
    note explains, so the UI shows 'no measurement', not a number."""
    m = metrics["LFM2.5-1.2B-Instruct-GGUF"]
    assert m.rate_implausible is True
    assert m.tokens_per_second == 1_000_000
    assert m.rate_note and "not a real rate" in m.rate_note
    assert "1 token" in m.rate_note  # explains WHY (last request produced 1 token)


def test_aevum_plausible_high_rate_is_kept(metrics):
    """Aevum genuinely runs ~229 tok/s (small model). That is under the guard,
    so it must be KEPT — the guard rejects 1e6 artifacts, not fast models."""
    m = metrics["Aevum-0.6B-Finetuned.Q4_K_M"]
    assert m.tokens_per_second == pytest.approx(229.3, abs=0.5)
    assert m.rate_implausible is False


def test_empty_or_garbage_yields_no_rows():
    assert parse_model_metrics(None) == {}
    assert parse_model_metrics("") == {}
    assert parse_model_metrics("# just a comment\nrandom junk\n") == {}


def test_absent_metric_is_none_not_zero():
    """A model line with no output_tokens_total series reads None, never 0 —
    the F-series absence-is-not-a-measurement rule."""
    text = (
        'lemonade_model_info{model_name="X",checkpoint="repo/x"} 1\n'
        'lemonade_model_requests_total{model_name="X",checkpoint="repo/x"} 3\n'
    )
    m = parse_model_metrics(text)["X"]
    assert m.requests_total == 3
    assert m.output_tokens_total is None
    assert m.tokens_per_second is None
    assert m.rate_implausible is False
