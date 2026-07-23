"""Rule 4 / F2 tests.

`120.0` and `0.5` appear in this file and in tests/fixtures/system-info.json.
Those are the ONLY two places they are allowed to exist (F2). There is a test
at the bottom that enforces it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from kblocallm.domain.pool import PoolWarning, extract_pool

FIXTURES = Path(__file__).parent.parent / "fixtures"
SRC = Path(__file__).parent.parent.parent / "src" / "kblocallm"


@pytest.fixture(scope="module")
def system_info() -> dict:
    """The raw capture from the Lemonade host, 2026-07-16 (F2 says pin it)."""
    return json.loads((FIXTURES / "system-info.json").read_text())


@pytest.fixture(scope="module")
def system_stats() -> dict:
    return json.loads((FIXTURES / "system-stats.json").read_text())


# -- the live payload ------------------------------------------------------


def test_the_pool_is_read_from_the_live_capture(system_info, system_stats):
    """The whole of F2 in one assertion: 120.0 from the nested amd_gpu entry,
    NOT 0.5 sitting beside it and NOT the 124.95 string above it."""
    pool = extract_pool(system_info, system_stats, gtt_enabled=True)
    assert pool.capacity_gb == 120.0
    assert pool.source == "system-info.devices.amd_gpu[0].virtual_mem_gb"
    assert pool.warning is None


def test_the_numerator_comes_from_the_other_endpoint(system_info, system_stats):
    """Two fields named vram_gb. The one on system-stats is usage; the one on
    system-info is the 0.5GB carve-out."""
    pool = extract_pool(system_info, system_stats, gtt_enabled=True)
    assert pool.used_gb == pytest.approx(56.42, abs=0.01)
    assert pool.used_gb != 0.5, "that would be the carve-out from the wrong endpoint"
    assert pool.percent == pytest.approx(47.0, abs=0.5)


def test_the_carve_out_is_never_the_capacity(system_info, system_stats):
    """The liar sits in the same object as the truth. Name it in a test so a
    future 'simplification' to `gpu["vram_gb"]` goes red instead of live."""
    gpu = system_info["devices"]["amd_gpu"][0]
    assert gpu["vram_gb"] == 0.5, "fixture drift: the carve-out moved"
    assert extract_pool(system_info, system_stats).capacity_gb != gpu["vram_gb"]


def test_physical_memory_string_is_never_the_capacity(system_info, system_stats):
    """124.95 GB is system RAM and a string. Close enough to 120 to look
    right, which is exactly what makes it dangerous."""
    assert system_info["Physical Memory"] == "124.95 GB"
    assert extract_pool(system_info, system_stats).capacity_gb == 120.0


def test_the_unavailable_nvidia_list_is_not_indexed(system_info):
    """An nvidia_gpu list with available:false sits next to amd_gpu. Blind
    indexing across the wrong list is the failure this guards."""
    assert system_info["devices"]["nvidia_gpu"][0]["available"] is False
    assert extract_pool(system_info).capacity_gb == 120.0


# -- device selection ------------------------------------------------------


def _info(amd_gpu, **extra):
    return {"devices": {"amd_gpu": amd_gpu, "nvidia_gpu": [{"available": False}]}, **extra}


def test_the_available_entry_is_selected_not_the_first():
    """Do not blind-index [0]. Here [0] is a stale unavailable device with a
    plausible-looking pool that would render a confident, wrong bar."""
    pool = extract_pool(_info([
        {"available": False, "virtual_mem_gb": 8.0},
        {"available": True, "virtual_mem_gb": 120.0},
    ]))
    assert pool.capacity_gb == 120.0
    assert pool.source.endswith("amd_gpu[1].virtual_mem_gb")


def test_no_available_gpu_reads_unknown_not_zero():
    pool = extract_pool(_info([{"available": False, "virtual_mem_gb": 8.0}]))
    assert pool.capacity_gb is None
    assert pool.percent is None
    assert pool.warning is PoolWarning.POOL_UNREADABLE


def test_multiple_available_gpus_refuse_a_denominator():
    """system-stats.vram_gb is one total; two pools have no single correct
    denominator. Refuse and say so rather than draw a bar against a guess."""
    pool = extract_pool(_info([
        {"available": True, "virtual_mem_gb": 120.0},
        {"available": True, "virtual_mem_gb": 24.0},
    ]))
    assert pool.capacity_gb is None
    assert pool.warning is PoolWarning.AMBIGUOUS_DEVICE


def test_a_dict_instead_of_a_list_still_reads():
    """This payload's shape was documented wrong once already. Tolerate the
    other shape; never guess the number."""
    pool = extract_pool({"devices": {"amd_gpu": {"available": True, "virtual_mem_gb": 120.0}}})
    assert pool.capacity_gb == 120.0


@pytest.mark.parametrize("payload", [None, {}, {"devices": None}, {"devices": {}},
                                     {"devices": {"amd_gpu": "nonsense"}}])
def test_junk_payloads_read_unknown_and_do_not_crash(payload):
    pool = extract_pool(payload)
    assert pool.capacity_gb is None and pool.percent is None
    assert pool.warning is PoolWarning.POOL_UNREADABLE


@pytest.mark.parametrize("bad", [None, 0, -1, "120", True])
def test_unusable_capacity_never_falls_back_to_the_carve_out(bad):
    """The tempting 'fix' is `or gpu["vram_gb"]`. That is the saga. The
    detail string names it so the next reader is warned in place."""
    pool = extract_pool(_info([{"available": True, "virtual_mem_gb": bad, "vram_gb": 0.5}]))
    assert pool.capacity_gb is None
    assert "carve-out" in pool.warning_detail


# -- the cross-check -------------------------------------------------------


SMALL_POOL = [{"available": True, "virtual_mem_gb": 0.5, "vram_gb": 0.5}]


def test_small_pool_plus_gtt_off_is_the_hard_warning():
    """The saga's signature, catchable in one glance. The message must name
    the fix, not just the symptom."""
    pool = extract_pool(_info(SMALL_POOL), gtt_enabled=False)
    assert pool.warning is PoolWarning.SMALL_POOL_GTT_DISABLED
    assert "enable_dgpu_gtt true" in pool.warning_detail
    # Still reports the real number. The pool IS 0.5 right now; the warning
    # explains why, it doesn't paper over it with the number we wish it was.
    assert pool.capacity_gb == 0.5


def test_small_pool_with_gtt_unreadable_says_so_rather_than_assuming():
    pool = extract_pool(_info(SMALL_POOL), gtt_enabled=None)
    assert pool.warning is PoolWarning.SMALL_POOL_GTT_UNKNOWN
    assert "could not read enable_dgpu_gtt" in pool.warning_detail


def test_small_pool_with_gtt_on_does_not_blame_gtt():
    """Sending the operator to flip a flag that is already on wastes the hour
    the warning was supposed to save."""
    pool = extract_pool(_info(SMALL_POOL), gtt_enabled=True)
    assert "BIOS memory split" in pool.warning_detail
    assert "set enable_dgpu_gtt true" not in pool.warning_detail


def test_a_healthy_pool_warns_about_nothing(system_info):
    assert extract_pool(system_info, gtt_enabled=True).warning is None


def test_gtt_flag_matches_the_live_params_capture():
    """The cross-check's input, straight off the box (/api/v1/params)."""
    params = json.loads((FIXTURES / "params.json").read_text())
    assert params["enable_dgpu_gtt"] is True


# -- F2's literal ban ------------------------------------------------------


def test_no_hardcoded_pool_constant_in_the_source_tree():
    """F2: '120.5 may appear in source ONLY as a test fixture.' The literal is
    the same bug class as the 80GB one, pre-loaded with today's right answer:
    it goes wrong the moment enable_dgpu_gtt flips, the box gets more RAM, or
    HELM points at a second machine.

    Parsed with `ast`, not grepped: the docs and this package's own comments
    discuss 120.0 constantly, and a text search cannot tell prose about the
    saga from a value that would recreate it. Only real numeric literals in
    real code count.
    """
    banned = {120.5, 120.0, 0.5}
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in banned:
                if isinstance(node.value, bool):
                    continue
                offenders.append(f"{path.name}:{node.lineno}: literal {node.value}")
    assert not offenders, (
        "Pool figures must be read from the API on every poll, never hardcoded "
        "(F2). 120.0 is today's right answer and 0.5 is the carve-out that "
        "caused the saga; neither belongs in source:\n" + "\n".join(offenders)
    )
