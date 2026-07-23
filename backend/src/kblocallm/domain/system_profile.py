"""The live system profile. Detected, never hardcoded.

HELM must be deployable to any box (Kevin is buying more Strix Halo machines),
so it cannot carry one machine's specs baked into code or prose. Everything
here is read live from `/api/v1/system-info` + `/api/v1/system-stats`, the same
source the dashboard's memory bar already trusts. Deploy HELM to a different
box and this reports THAT box.

Two consumers:
  - the Operations "Current system setup" panel, so the operator can see what
    HELM detected;
  - the Model Advisor's context, so its hardware reasoning ("does it fit", "how
    fast will it decode") grounds in the DETECTED pool and engines, not in a
    knowledge file's example numbers.

Pure and table-tested, off the real captured system-info. The pool figure comes
from `extract_pool` so there is one source of truth for the number that caused
the 80GB saga.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pool import extract_pool


@dataclass(frozen=True)
class BackendInfo:
    recipe: str
    installed: list[str] = field(default_factory=list)     # backends ready to use
    installable: list[str] = field(default_factory=list)   # available to install


@dataclass(frozen=True)
class SystemProfile:
    #: Processor string, e.g. "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S".
    processor: str | None = None
    cpu_cores: int | None = None
    cpu_threads: int | None = None

    gpu_name: str | None = None
    gpu_family: str | None = None       # e.g. gfx1151
    #: THE memory pool the GPU can use (the denominator). Detected, in GB.
    gpu_pool_gb: float | None = None
    gpu_pool_source: str | None = None  # the field path it came from

    npu_available: bool = False
    npu_family: str | None = None       # e.g. XDNA2
    npu_tops: int | None = None
    npu_power_mode: str | None = None

    #: Total physical RAM as the box reports it (a string, e.g. "124.95 GB").
    physical_memory: str | None = None
    #: Current GPU memory in use (numerator), from system-stats.
    used_gb: float | None = None

    os_version: str | None = None
    lemonade_version: str | None = None

    storage_path: str | None = None
    storage_free_gb: float | None = None
    storage_total_gb: float | None = None

    #: Inference backends the box exposes, with install state. This is what a
    #: new box would differ on, so it is detected, not assumed.
    backends: list[BackendInfo] = field(default_factory=list)

    #: True when we could not read system-info at all; the panel says so rather
    #: than showing a fabricated machine.
    detected: bool = True


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _bytes_to_gb(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return round(value / (1024 ** 3), 1)
    return None


def extract_system_profile(
    system_info: dict[str, Any] | None,
    system_stats: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> SystemProfile:
    """Build the profile from live payloads. Any may be None; fields degrade
    to None rather than to an invented value."""
    if not isinstance(system_info, dict):
        return SystemProfile(detected=False)

    devices = system_info.get("devices") if isinstance(system_info.get("devices"), dict) else {}
    cpu = devices.get("cpu") if isinstance(devices.get("cpu"), dict) else {}
    npu = devices.get("amd_npu") if isinstance(devices.get("amd_npu"), dict) else {}

    gtt = (params or {}).get("enable_dgpu_gtt") if isinstance(params, dict) else None
    pool = extract_pool(system_info, system_stats,
                        gtt_enabled=gtt if isinstance(gtt, bool) else None)

    # The available AMD GPU entry, for name/family (the pool number itself
    # comes from extract_pool, the single source of truth).
    gpu_name = gpu_family = None
    amd = devices.get("amd_gpu")
    if isinstance(amd, dict):
        amd = [amd]
    if isinstance(amd, list):
        for entry in amd:
            if isinstance(entry, dict) and entry.get("available") is True:
                gpu_name = entry.get("name")
                gpu_family = entry.get("family")
                break

    storage = system_info.get("model_storage")
    storage = storage if isinstance(storage, dict) else {}

    return SystemProfile(
        processor=system_info.get("Processor"),
        cpu_cores=_int(cpu.get("cores")),
        cpu_threads=_int(cpu.get("threads")),
        gpu_name=gpu_name,
        gpu_family=gpu_family,
        gpu_pool_gb=pool.capacity_gb,
        gpu_pool_source=pool.source,
        npu_available=npu.get("available") is True,
        npu_family=npu.get("family"),
        npu_tops=_int(npu.get("tops_max_int")),
        npu_power_mode=npu.get("power_mode"),
        physical_memory=system_info.get("Physical Memory"),
        used_gb=pool.used_gb,
        os_version=system_info.get("OS Version"),
        lemonade_version=(health or {}).get("version") if isinstance(health, dict) else None,
        storage_path=storage.get("path"),
        storage_free_gb=_bytes_to_gb(storage.get("free_bytes")),
        storage_total_gb=_bytes_to_gb(storage.get("total_bytes")),
        backends=_backends(system_info.get("recipes")),
        detected=True,
    )


def _backends(recipes: Any) -> list[BackendInfo]:
    """Which inference backends this box exposes, installed vs installable.

    Straight from system-info's `recipes` block, so a box with different
    engines installed reports differently — nothing assumed."""
    if not isinstance(recipes, dict):
        return []
    out: list[BackendInfo] = []
    for recipe, meta in sorted(recipes.items()):
        if not isinstance(meta, dict):
            continue
        backends = meta.get("backends")
        if not isinstance(backends, dict):
            continue
        installed, installable = [], []
        for name, b in backends.items():
            if not isinstance(b, dict):
                continue
            state = b.get("state")
            if state == "installed" or (b.get("action") == "" and state not in ("unsupported",)):
                installed.append(name)
            elif state == "installable":
                installable.append(name)
        if installed or installable:
            out.append(BackendInfo(recipe=recipe, installed=sorted(installed),
                                   installable=sorted(installable)))
    return out


def advisor_hardware_context(profile: SystemProfile) -> dict[str, Any]:
    """The compact, LIVE hardware facts to hand the advisor, so it reasons about
    THIS box. The advisor is told (in its instructions) to prefer these over any
    example numbers in the knowledge file."""
    if not profile.detected:
        return {"status": "undetected",
                "note": "HELM could not read this box's hardware; do not assume specs."}
    return {
        "processor": profile.processor,
        "cpu_cores": profile.cpu_cores,
        "cpu_threads": profile.cpu_threads,
        "gpu": profile.gpu_name,
        "gpu_family": profile.gpu_family,
        "gpu_memory_pool_gb": profile.gpu_pool_gb,
        "gpu_memory_in_use_gb": profile.used_gb,
        "physical_memory": profile.physical_memory,
        "npu": (f"{profile.npu_family} (~{profile.npu_tops} TOPS)"
                if profile.npu_available else "none detected"),
        "note": "These are DETECTED from this box right now. Ground all fit and speed reasoning "
                "in these numbers, not in example hardware from the knowledge file.",
    }
