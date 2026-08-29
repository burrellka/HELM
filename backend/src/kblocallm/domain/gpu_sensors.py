"""Local GPU thermals — temperature and power, read from the kernel's hwmon.

Lemonade's `/api/v1/system-stats` gives us `gpu_percent`, but on this class of
hardware that gauge is stuck at 100% — a documented amdgpu/ROCm bug on gfx1151
(the Radeon 8060S in Strix Halo): amd-smi reports N/A for the monitoring
metrics while the kernel still exposes correct values via sysfs (ROCm #6035),
alongside gfx1151 power-state / VRAM-reporting bugs (#5750).

So we do NOT remove the gauge — it is correct for everyone whose driver reports
honestly. Instead we read two things the kernel gets right on *every* AMD GPU,
straight from hwmon:

    /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input     # millidegrees C
    /sys/class/drm/card*/device/hwmon/hwmon*/power1_average  # microwatts

They are genuinely more useful than a percentage anyway: idle-vs-working is
unambiguous, and thermal visibility matters because sustained-load thermal
reboots are documented on this box.

And when the gauge is *clearly* lying — pinned at 100% while temp and power both
say idle — we annotate it (like the NPU gauge), so the number self-labels as
unreliable. The annotation is driven by the live readings, not by a hardware ID,
so the day amdgpu is fixed upstream and the gauge reports honestly, the note
simply stops appearing. No feature removed, no regression for other users.

I/O lives in `read_gpu_sensors` (root injectable, so tests point it at a fixture
tree); the annotation rule is the pure, table-tested `interpret_gpu_gauge`.
On any box without these sysfs nodes — HELM not co-located with the GPU, or a
non-Linux host — every field degrades to None. Never a faked thermal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A stuck gauge is pinned high; a genuinely-busy GPU that happens to read 100%
# will not also be cool AND sipping power. These two together are what make the
# "it's lying" call safe — under real sustained load either would blow past.
_STUCK_PERCENT = 99.0
_IDLE_TEMP_C = 60.0
_IDLE_POWER_W = 50.0

# Only real DRM cards (card0, card1, ...). The `card*` glob also matches
# connector subdirs like `card0-eDP-1`, which point at the same device — this
# keeps us to one hwmon per physical card.
_CARD_RE = re.compile(r"^card\d+$")

_GAUGE_UNRELIABLE_NOTE = (
    "GPU% may be unreliable on this hardware; temp/power indicate idle. "
    "Known upstream issue on gfx1151 (ROCm #6035, #5750)."
)


@dataclass(frozen=True)
class GpuSensors:
    #: GPU temperature in °C, from hwmon temp1_input. None if unreadable.
    temp_c: float | None = None
    #: GPU average power draw in watts, from hwmon power1_average. None if
    #: unreadable (some cards expose temp but not power1_average).
    power_w: float | None = None
    #: The hwmon directory the readings came from, for the "where did this
    #: number come from" ethos. None when nothing was read.
    source: str | None = None


def interpret_gpu_gauge(
    gpu_percent: float | None,
    temp_c: float | None,
    power_w: float | None,
) -> str | None:
    """The annotation for a stuck utilization gauge, or None.

    Returns the note only when we can actually *prove* the gauge is lying: it is
    pinned high AND we have real thermals AND both say idle. Missing any of the
    three (no reading, or a plausibly-busy GPU) returns None — we never annotate
    on suspicion, only on evidence.
    """
    if gpu_percent is None or gpu_percent < _STUCK_PERCENT:
        return None
    if temp_c is None or power_w is None:
        return None
    if temp_c < _IDLE_TEMP_C and power_w < _IDLE_POWER_W:
        return _GAUGE_UNRELIABLE_NOTE
    return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_name(hwmon: Path) -> str | None:
    try:
        return hwmon.joinpath("name").read_text().strip()
    except OSError:
        return None


def read_gpu_sensors(drm_root: Path = Path("/sys/class/drm")) -> GpuSensors:
    """Read GPU temp/power from hwmon under `drm_root`. Pure degradation to
    None on any box where the nodes are absent or unreadable."""
    try:
        cards = sorted(p for p in drm_root.iterdir() if _CARD_RE.match(p.name))
    except OSError:
        return GpuSensors()

    # Prefer an amdgpu hwmon that yields a temperature. Fall back to the first
    # hwmon that reads a temp at all, so a differently-named AMD driver still
    # reports rather than showing nothing.
    fallback: GpuSensors | None = None
    for card in cards:
        hwmon_root = card / "device" / "hwmon"
        try:
            hwmons = sorted(hwmon_root.iterdir())
        except OSError:
            continue
        for hwmon in hwmons:
            temp_raw = _read_int(hwmon / "temp1_input")
            if temp_raw is None:
                continue
            power_raw = _read_int(hwmon / "power1_average")
            reading = GpuSensors(
                temp_c=round(temp_raw / 1000, 1),
                power_w=round(power_raw / 1_000_000, 1) if power_raw is not None else None,
                source=str(hwmon),
            )
            if _read_name(hwmon) == "amdgpu":
                return reading
            if fallback is None:
                fallback = reading

    return fallback or GpuSensors()
