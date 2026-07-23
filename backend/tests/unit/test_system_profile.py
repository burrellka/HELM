"""System profile tests — everything DETECTED off the real captured payloads,
nothing hardcoded, so HELM ports to any box."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kblocallm.domain.system_profile import advisor_hardware_context, extract_system_profile

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def info():
    return json.loads((FIXTURES / "system-info.json").read_text())


@pytest.fixture(scope="module")
def stats():
    return json.loads((FIXTURES / "system-stats.json").read_text())


@pytest.fixture(scope="module")
def params():
    return json.loads((FIXTURES / "params.json").read_text())


def test_the_profile_is_read_live_off_this_box(info, stats, params):
    p = extract_system_profile(info, stats, health={"version": "10.10.0"}, params=params)
    assert p.detected is True
    assert "RYZEN AI MAX+ 395" in p.processor
    assert (p.cpu_cores, p.cpu_threads) == (16, 32)
    assert p.gpu_family == "gfx1151"
    # THE pool, from the one true source — not a hardcoded 128 or 120.5.
    assert p.gpu_pool_gb == 120.0
    assert p.gpu_pool_source == "system-info.devices.amd_gpu[0].virtual_mem_gb"
    assert p.used_gb == pytest.approx(56.42, abs=0.01)
    assert p.physical_memory == "124.95 GB"
    assert p.lemonade_version == "10.10.0"


def test_the_npu_is_detected_not_assumed(info):
    p = extract_system_profile(info)
    assert p.npu_available is True
    assert p.npu_family == "XDNA2"
    assert p.npu_tops == 58


def test_backends_are_detected_with_install_state(info):
    p = extract_system_profile(info)
    recipes = {b.recipe for b in p.backends}
    assert "llamacpp" in recipes and "flm" in recipes
    llamacpp = next(b for b in p.backends if b.recipe == "llamacpp")
    # It reports what is actually installed vs installable, per box.
    assert llamacpp.installed or llamacpp.installable


def test_storage_is_detected():
    info = {"devices": {}, "model_storage": {
        "path": "/var/lib/lemonade", "free_bytes": 1574828630016, "total_bytes": 2014574526464}}
    p = extract_system_profile(info)
    assert p.storage_path == "/var/lib/lemonade"
    assert p.storage_free_gb == pytest.approx(1466.7, abs=1)


def test_a_different_box_reports_differently():
    """The portability guarantee: change the payload, the profile changes. No
    box's specs are baked in."""
    other = {
        "Processor": "Some Other CPU", "Physical Memory": "64.00 GB",
        "devices": {
            "cpu": {"cores": 8, "threads": 16},
            "amd_gpu": [{"available": True, "family": "gfx9999", "virtual_mem_gb": 48.0, "vram_gb": 0.5}],
            "amd_npu": {"available": False},
        },
    }
    p = extract_system_profile(other)
    assert p.gpu_pool_gb == 48.0          # not 120
    assert p.gpu_family == "gfx9999"
    assert p.npu_available is False
    assert (p.cpu_cores, p.cpu_threads) == (8, 16)


def test_undetected_says_so_rather_than_inventing():
    p = extract_system_profile(None)
    assert p.detected is False
    ctx = advisor_hardware_context(p)
    assert ctx["status"] == "undetected"
    assert "do not assume" in ctx["note"]


def test_advisor_context_carries_the_live_pool_not_a_baked_number(info, stats):
    ctx = advisor_hardware_context(extract_system_profile(info, stats))
    assert ctx["gpu_memory_pool_gb"] == 120.0
    assert "DETECTED from this box" in ctx["note"]
    assert "not in example hardware" in ctx["note"]
