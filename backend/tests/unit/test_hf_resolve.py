"""Repo resolution tests (14c §1.8). A wrong guess is a confident lie, so the
tests are mostly about what it REFUSES to resolve."""
from __future__ import annotations

import pytest

from kblocallm.domain.hf_resolve import RepoConfidence, resolve_repo


@pytest.mark.parametrize("checkpoint,expected", [
    ("unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M", "unsloth/gemma-4-26B-A4B-it-GGUF"),
    ("NousResearch/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M", "NousResearch/Hermes-3-Llama-3.1-8B-GGUF"),
    ("unsloth/Qwen3.5-4B-MTP-GGUF:Qwen3.5-4B-UD-Q4_K_XL.gguf", "unsloth/Qwen3.5-4B-MTP-GGUF"),
])
def test_explicit_repo_ids_resolve(checkpoint, expected):
    r = resolve_repo(checkpoint)
    assert r.confidence is RepoConfidence.EXPLICIT
    assert r.repo_id == expected


def test_a_local_path_resolves_to_no_record_not_a_guess():
    """Aevum is Kevin's own finetune at a local path. The honest answer is 'no
    external record', never an invented org/Aevum repo."""
    r = resolve_repo("/opt/models/models/Aevum-0.6B-Finetuned.Q4_K_M.gguf")
    assert r.confidence is RepoConfidence.NONE
    assert "local file path" in r.reason


def test_a_bare_flm_tag_is_not_fuzzy_matched():
    """gemma4-it:e2b is not an org/repo. Guessing a repo from it risks the
    wrong card."""
    r = resolve_repo("gemma4-it:e2b")
    assert r.confidence is RepoConfidence.NONE


def test_a_windows_path_is_treated_as_local():
    assert resolve_repo(r"C:\models\thing.gguf").confidence is RepoConfidence.NONE


def test_no_checkpoint_declines_cleanly():
    r = resolve_repo(None)
    assert r.confidence is RepoConfidence.NONE
    assert "does not guess" in r.reason


def test_a_three_segment_path_is_not_a_repo():
    """org/repo/extra is not a valid two-segment repo id."""
    assert resolve_repo("a/b/c").confidence is RepoConfidence.NONE
