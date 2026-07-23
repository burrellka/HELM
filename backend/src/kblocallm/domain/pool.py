"""The memory pool. Spec Rule 4, review F2, handoff §3.

This module exists because a wrong number here cost months. The short history:
an app-layer bug read the 512MB BIOS carve-out instead of the 120GB GTT pool,
reported a phantom ceiling, and everything downstream reasoned from it.

The trap is still sitting in the payload, and it is nastier than the docs
made it sound. `system-info.devices.amd_gpu[0]` carries BOTH numbers, next to
each other, and the wrong one has the friendlier name:

    {"available": true, "family": "gfx1151",
     "virtual_mem_gb": 120.0,   <- the pool. The denominator.
     "vram_gb": 0.5}            <- the BIOS carve-out. THE LIAR.

And there are two more decoys:
  - top-level `"Physical Memory": "124.95 GB"` — a STRING, and system RAM. It
    is close enough to 120 to look right and it is the wrong denominator.
  - `vram_gb` on the DIFFERENT /api/v1/system-stats endpoint — THAT one is the
    numerator (current usage). Two fields named `vram_gb`, two endpoints, two
    meanings.

So: no `120.5` literal anywhere in this package (it lives in the fixture, and
only there), read the pool per poll, and cross-check `enable_dgpu_gtt` — the
pairing "pool reads small AND gtt is off" is the saga's actual signature, and
HELM can catch in one glance what took months.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Below this, the reported pool cannot hold any model worth loading and is
#: almost certainly the BIOS carve-out (0.5GB on this box) rather than the GTT
#: pool. An absolute floor, deliberately: a fraction-of-RAM heuristic would
#: false-positive on a real discrete GPU (24GB card, 128GB host) and cry wolf.
SMALL_POOL_GB = 16.0


class PoolWarning(str, Enum):
    #: The saga's exact signature. Actionable: `lemonade config set
    #: enable_dgpu_gtt true`.
    SMALL_POOL_GTT_DISABLED = "small_pool_gtt_disabled"
    #: Small pool, and we could not read the flag to confirm why.
    SMALL_POOL_GTT_UNKNOWN = "small_pool_gtt_unknown"
    #: We could not find the pool at all. The bar renders `unknown`; it does
    #: NOT render against a guess.
    POOL_UNREADABLE = "pool_unreadable"
    #: More than one available GPU. A single denominator would be a fiction.
    AMBIGUOUS_DEVICE = "ambiguous_device"


@dataclass(frozen=True)
class MemoryPool:
    #: Capacity (denominator). None means unknown — render `unknown`, never 0,
    #: and never a remembered constant.
    capacity_gb: float | None
    #: Current usage (numerator), from system-stats.
    used_gb: float | None
    #: The field path we actually read, e.g.
    #: "system-info.devices.amd_gpu[0].virtual_mem_gb". Shown in the UI
    #: tooltip. When the next person asks "where does 120 come from?", the
    #: answer should be on screen, not in a code archaeology session.
    source: str | None = None
    warning: PoolWarning | None = None
    warning_detail: str | None = None
    #: Verbatim from /api/v1/params. None = we could not read it.
    gtt_enabled: bool | None = None

    @property
    def percent(self) -> float | None:
        """Usage as a percentage of the real pool. None whenever either half
        is unknown — a bar drawn against a guessed denominator is exactly the
        thing this module exists to prevent."""
        if self.capacity_gb is None or self.used_gb is None or self.capacity_gb <= 0:
            return None
        return (self.used_gb / self.capacity_gb) * 100.0


def _available_gpus(system_info: dict[str, Any]) -> list[tuple[int, dict]]:
    """AMD GPU entries reporting available: true, with their indices.

    `amd_gpu` is a LIST and the doc's original `devices.virtual_mem_gb` path
    was simply wrong. An `nvidia_gpu` list sits beside it with
    `available: false`, which is why blind-indexing `[0]` on the wrong list is
    a real hazard rather than a hypothetical one.
    """
    devices = system_info.get("devices")
    if not isinstance(devices, dict):
        return []
    entries = devices.get("amd_gpu")
    # The spec said list; the box says list. Accept a bare dict too — this
    # payload's shape has already been documented wrong once, and tolerating
    # the other shape costs two lines while guessing a number costs months.
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return []
    return [
        (i, e) for i, e in enumerate(entries)
        if isinstance(e, dict) and e.get("available") is True
    ]


def extract_pool(
    system_info: dict[str, Any] | None,
    system_stats: dict[str, Any] | None = None,
    gtt_enabled: bool | None = None,
) -> MemoryPool:
    """Build the memory pool view. The only place the denominator is decided.

    `gtt_enabled` comes from /api/v1/params (`enable_dgpu_gtt`); pass None when
    it could not be read, and the cross-check degrades to "small pool, cause
    unconfirmed" instead of asserting a cause it does not know.
    """
    used = _used_gb(system_stats)

    if not isinstance(system_info, dict):
        return MemoryPool(
            capacity_gb=None, used_gb=used, gtt_enabled=gtt_enabled,
            warning=PoolWarning.POOL_UNREADABLE,
            warning_detail="No system-info payload to read the pool from.",
        )

    gpus = _available_gpus(system_info)

    if not gpus:
        return MemoryPool(
            capacity_gb=None, used_gb=used, gtt_enabled=gtt_enabled,
            warning=PoolWarning.POOL_UNREADABLE,
            warning_detail=(
                "No AMD GPU reporting available:true in "
                "system-info.devices.amd_gpu. Cannot determine the memory pool."
            ),
        )

    if len(gpus) > 1:
        # One vram_gb total over several pools has no correct denominator.
        # Refuse rather than pick one and draw a confident bar against it.
        return MemoryPool(
            capacity_gb=None, used_gb=used, gtt_enabled=gtt_enabled,
            warning=PoolWarning.AMBIGUOUS_DEVICE,
            warning_detail=(
                f"{len(gpus)} available AMD GPUs reported; system-stats.vram_gb "
                f"is a single total, so no single pool is the right denominator. "
                f"HELM needs teaching about multi-GPU before it can show this."
            ),
        )

    index, gpu = gpus[0]
    source = f"system-info.devices.amd_gpu[{index}].virtual_mem_gb"
    capacity = gpu.get("virtual_mem_gb")

    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or capacity <= 0:
        return MemoryPool(
            capacity_gb=None, used_gb=used, source=source, gtt_enabled=gtt_enabled,
            warning=PoolWarning.POOL_UNREADABLE,
            warning_detail=(
                f"{source} is {capacity!r}, not a usable number. NOT falling back "
                f"to vram_gb in the same object — that is the 0.5GB BIOS "
                f"carve-out and reading it is how the 80GB saga started."
            ),
        )

    capacity = float(capacity)
    pool = MemoryPool(
        capacity_gb=capacity, used_gb=used, source=source, gtt_enabled=gtt_enabled,
    )

    if capacity >= SMALL_POOL_GB:
        return pool

    # --- the saga's signature ---------------------------------------------
    if gtt_enabled is False:
        return _with_warning(
            pool, PoolWarning.SMALL_POOL_GTT_DISABLED,
            f"The GPU memory pool reads {capacity:.1f} GB and enable_dgpu_gtt is "
            f"FALSE. That pairing is the known cause: without GTT the server sees "
            f"only the BIOS carve-out instead of the unified pool. Fix on "
            f"the Lemonade host with `lemonade config set enable_dgpu_gtt true`, then "
            f"restart lemond.",
        )
    if gtt_enabled is None:
        return _with_warning(
            pool, PoolWarning.SMALL_POOL_GTT_UNKNOWN,
            f"The GPU memory pool reads {capacity:.1f} GB, which is too small to "
            f"hold a useful model. HELM could not read enable_dgpu_gtt from "
            f"/api/v1/params to confirm the cause — check it on the box.",
        )
    # Small pool but GTT is genuinely on: a different problem, and saying
    # "turn on GTT" here would send the operator down the wrong path.
    return _with_warning(
        pool, PoolWarning.SMALL_POOL_GTT_UNKNOWN,
        f"The GPU memory pool reads {capacity:.1f} GB even though enable_dgpu_gtt "
        f"is true. That is not the usual GTT cause; check the BIOS memory split "
        f"and lemond's startup log for the 'Largest memory pool:' line.",
    )


def _with_warning(pool: MemoryPool, warning: PoolWarning, detail: str) -> MemoryPool:
    return MemoryPool(
        capacity_gb=pool.capacity_gb, used_gb=pool.used_gb, source=pool.source,
        gtt_enabled=pool.gtt_enabled, warning=warning, warning_detail=detail,
    )


def _used_gb(system_stats: dict[str, Any] | None) -> float | None:
    """The NUMERATOR: `vram_gb` on /api/v1/system-stats.

    Not to be confused with `vram_gb` on system-info, which is the 0.5GB
    carve-out. Same name, different endpoint, different meaning; this comment
    is the only thing standing between the next reader and that mistake.
    """
    if not isinstance(system_stats, dict):
        return None
    used = system_stats.get("vram_gb")
    if isinstance(used, bool) or not isinstance(used, (int, float)) or used < 0:
        return None
    return float(used)
