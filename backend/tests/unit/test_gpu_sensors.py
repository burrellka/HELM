"""Tests for the local GPU thermals reader and the stuck-gauge annotation.

The annotation logic is pure and exhaustively table-tested. The hwmon reader is
exercised against a fixture sysfs tree built in a tmp dir, so we prove the glob,
the unit conversions, the amdgpu preference, and graceful degradation without
touching a real /sys.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kblocallm.domain.gpu_sensors import (
    GpuSensors,
    interpret_gpu_gauge,
    read_gpu_sensors,
)


# --- interpret_gpu_gauge: the annotation only on proof --------------------

def test_stuck_gauge_annotated_when_pinned_and_idle():
    note = interpret_gpu_gauge(100.0, 45.0, 18.0)
    assert note is not None
    assert "gfx1151" in note and "6035" in note


def test_no_annotation_when_gauge_low():
    # A gauge reading a normal value is trusted, whatever the thermals.
    assert interpret_gpu_gauge(30.0, 45.0, 18.0) is None


def test_no_annotation_when_genuinely_busy():
    # Pinned high AND hot/drawing power = really working, not a stuck gauge.
    assert interpret_gpu_gauge(100.0, 78.0, 130.0) is None


def test_no_annotation_when_hot_but_low_power():
    # Only one idle signal is not proof; we never annotate on suspicion.
    assert interpret_gpu_gauge(100.0, 70.0, 10.0) is None


def test_no_annotation_when_low_power_but_no_temp():
    assert interpret_gpu_gauge(100.0, None, 5.0) is None


def test_no_annotation_without_any_thermals():
    # No hwmon (HELM off-box) -> we cannot prove anything, so no note.
    assert interpret_gpu_gauge(100.0, None, None) is None


def test_no_annotation_when_percent_missing():
    assert interpret_gpu_gauge(None, 40.0, 10.0) is None


def test_thresholds_are_strict_lower_bounds():
    # Exactly at the idle ceilings is NOT "clearly idle": no annotation.
    assert interpret_gpu_gauge(100.0, 60.0, 20.0) is None      # temp == 60
    assert interpret_gpu_gauge(100.0, 40.0, 50.0) is None      # power == 50
    # Just under both -> annotated.
    assert interpret_gpu_gauge(100.0, 59.9, 49.9) is not None


def test_percent_boundary():
    # 99 is the stuck threshold (inclusive); 98.9 is trusted.
    assert interpret_gpu_gauge(99.0, 40.0, 10.0) is not None
    assert interpret_gpu_gauge(98.9, 40.0, 10.0) is None


# --- read_gpu_sensors: fixture sysfs tree --------------------------------

def _make_hwmon(root: Path, card: str, hwmon: str, *,
                name: str | None, temp: str | None, power: str | None) -> None:
    d = root / card / "device" / "hwmon" / hwmon
    d.mkdir(parents=True, exist_ok=True)
    if name is not None:
        (d / "name").write_text(name)
    if temp is not None:
        (d / "temp1_input").write_text(temp)
    if power is not None:
        (d / "power1_average").write_text(power)


def test_reads_temp_and_power_with_unit_conversion(tmp_path: Path):
    _make_hwmon(tmp_path, "card0", "hwmon0",
                name="amdgpu", temp="52000", power="18500000")
    s = read_gpu_sensors(tmp_path)
    assert s.temp_c == 52.0        # 52000 millidegrees -> 52.0 C
    assert s.power_w == 18.5       # 18_500_000 microwatts -> 18.5 W
    assert s.source is not None and "hwmon0" in s.source


def test_temp_without_power_degrades_power_only(tmp_path: Path):
    _make_hwmon(tmp_path, "card0", "hwmon0",
                name="amdgpu", temp="61000", power=None)
    s = read_gpu_sensors(tmp_path)
    assert s.temp_c == 61.0
    assert s.power_w is None


def test_connector_subdirs_are_ignored(tmp_path: Path):
    # Only a connector dir exists (card0-eDP-1), no real card -> nothing read.
    _make_hwmon(tmp_path, "card0-eDP-1", "hwmon9",
                name="amdgpu", temp="99000", power="99000000")
    s = read_gpu_sensors(tmp_path)
    assert s == GpuSensors()


def test_prefers_amdgpu_over_other_driver(tmp_path: Path):
    # A non-amdgpu card comes first alphabetically but must not win.
    _make_hwmon(tmp_path, "card0", "hwmon0",
                name="acpitz", temp="40000", power=None)
    _make_hwmon(tmp_path, "card1", "hwmon1",
                name="amdgpu", temp="55000", power="20000000")
    s = read_gpu_sensors(tmp_path)
    assert s.temp_c == 55.0 and s.power_w == 20.0


def test_falls_back_to_first_temp_when_no_amdgpu(tmp_path: Path):
    _make_hwmon(tmp_path, "card0", "hwmon0",
                name="some_gpu", temp="47000", power="12000000")
    s = read_gpu_sensors(tmp_path)
    assert s.temp_c == 47.0 and s.power_w == 12.0


def test_missing_root_degrades_to_none(tmp_path: Path):
    s = read_gpu_sensors(tmp_path / "does-not-exist")
    assert s == GpuSensors()


def test_garbage_temp_is_ignored(tmp_path: Path):
    _make_hwmon(tmp_path, "card0", "hwmon0",
                name="amdgpu", temp="not-a-number", power="10000000")
    s = read_gpu_sensors(tmp_path)
    # Unparseable temp means this hwmon yields nothing; no card left -> all None.
    assert s == GpuSensors()
