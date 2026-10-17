"""Tests for HF GGUF grouping — split shards, subdirs, companions."""
from __future__ import annotations

from kblocallm.domain.hf_quants import group_gguf_files, is_companion

GB = 1_000_000_000


def test_companion_detection():
    assert is_companion("mmproj-Model-BF16.gguf")
    assert is_companion("IQ3_XXS/mmproj-F16.gguf")
    assert is_companion("mtp-gemma-4-26B.gguf")
    assert is_companion("model.imatrix.gguf")
    assert not is_companion("IQ3_XXS/Model-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
    assert not is_companion("Model-Q4_K_M.gguf")


def test_single_file_quant_is_one_group():
    groups = group_gguf_files([("Model-Q4_K_M.gguf", 16 * GB)])
    assert len(groups) == 1
    g = groups[0]
    assert g.primary == "Model-Q4_K_M.gguf"
    assert g.shards == ["Model-Q4_K_M.gguf"]
    assert not g.is_split
    assert g.size_bytes == 16 * GB


def test_split_shards_collapse_to_one_group_with_summed_size():
    groups = group_gguf_files([
        ("IQ3_XXS/M-IQ3_XXS-00001-of-00002.gguf", 47 * GB),
        ("IQ3_XXS/M-IQ3_XXS-00002-of-00002.gguf", 28 * GB),
    ])
    assert len(groups) == 1
    g = groups[0]
    assert g.is_split
    assert g.primary.endswith("00001-of-00002.gguf")   # register on shard 1
    assert len(g.shards) == 2
    assert g.size_bytes == 75 * GB                      # summed, not one shard
    assert g.subdir == "IQ3_XXS"


def test_shards_grouped_even_when_listed_out_of_order():
    groups = group_gguf_files([
        ("M-00002-of-00002.gguf", 10 * GB),
        ("M-00001-of-00002.gguf", 20 * GB),
    ])
    assert len(groups) == 1
    assert groups[0].primary == "M-00001-of-00002.gguf"   # lowest index wins
    assert groups[0].shards == ["M-00001-of-00002.gguf", "M-00002-of-00002.gguf"]


def test_companions_dropped_mmproj_and_mtp():
    groups = group_gguf_files([
        ("Q4_K_M/weights-Q4_K_M.gguf", 16 * GB),
        ("mmproj-BF16.gguf", 1 * GB),
        ("mtp-draft.gguf", GB // 4),
    ])
    assert len(groups) == 1
    assert groups[0].primary == "Q4_K_M/weights-Q4_K_M.gguf"


def test_unknown_shard_size_makes_total_unknown_not_partial():
    # A confident 47GB total when a 28GB shard size is missing would be a lie.
    groups = group_gguf_files([
        ("M-00001-of-00002.gguf", 47 * GB),
        ("M-00002-of-00002.gguf", None),
    ])
    assert groups[0].size_bytes is None


def test_different_quants_in_subdirs_stay_separate():
    groups = group_gguf_files([
        ("Q2_0/M-Q2_0-00001-of-00002.gguf", 37 * GB),
        ("Q2_0/M-Q2_0-00002-of-00002.gguf", 28 * GB),
        ("IQ3_S/M-IQ3_S-00001-of-00002.gguf", 54 * GB),
        ("IQ3_S/M-IQ3_S-00002-of-00002.gguf", 28 * GB),
    ])
    assert len(groups) == 2
    # Sorted ascending by total: Q2_0 (65) before IQ3_S (82).
    assert groups[0].subdir == "Q2_0" and groups[0].size_bytes == 65 * GB
    assert groups[1].subdir == "IQ3_S" and groups[1].size_bytes == 82 * GB


def test_ista_flash_next_repo_yields_four_quants():
    # The real repo layout that showed "no quants, card only": 4 quants x 2
    # shards in subdirs + one root mmproj companion.
    entries = []
    for d, s1 in [("Q2_0", 37), ("IQ2_XS", 39), ("IQ3_XXS", 47), ("IQ3_S", 54)]:
        entries.append((f"{d}/Qwen-GSQ-RCO-{d}-00001-of-00002.gguf", s1 * GB))
        entries.append((f"{d}/Qwen-GSQ-RCO-{d}-00002-of-00002.gguf", 28 * GB))
    entries.append(("mmproj-Qwen-BF16.gguf", 1 * GB))
    groups = group_gguf_files(entries)
    assert len(groups) == 4
    assert all(g.is_split and len(g.shards) == 2 for g in groups)
    assert [g.subdir for g in groups] == ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"]


def test_non_gguf_ignored():
    groups = group_gguf_files([("README.md", 100), ("config.json", 50)])
    assert groups == []
