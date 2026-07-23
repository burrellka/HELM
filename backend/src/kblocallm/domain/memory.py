"""Per-model memory attribution by delta. Spec §5.5(b), review F10.

**Lemonade exposes no per-model memory.** `system-stats.vram_gb` is a total;
there is no `memory_mb` on the health record. So the only available technique
is: snapshot the total, load the model, snapshot again, attribute the
difference.

**And the delta is dirtier than it sounds (F10).** All of these produce a
confidently wrong per-model number:
  - LRU eviction fires from the very load being measured, so the delta can
    include a NEGATIVE contribution from whatever the router dropped to make
    room.
  - Other clients load concurrently — the CLI, KAIROS, a second HELM tab.
  - The nuclear option can fire mid-measurement and evict everything.

So the rule here is: capture the full model composition before and after, and
**invalidate the measurement if anything changed except the target
appearing**. Show nothing rather than a number we know is polluted. An
"estimated" marker on a number corrupted by an eviction is not honesty; it is
a disclaimer on a lie.

Option 2 from the spec (`pid: host` + /proc) is rejected on
privilege-escalation grounds, not tidiness: it hands the container the host
PID namespace — visibility into and signal capability against every process on
the Lemonade host. That is not a coupling trade-off, it is a live risk on a control
plane.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Invalid(str, Enum):
    #: Something other than the target appeared or vanished mid-measurement.
    COMPOSITION_CHANGED = "composition_changed"
    #: The target never showed up; the load did not take effect.
    TARGET_ABSENT = "target_absent"
    #: A reading was missing or unusable.
    NO_READING = "no_reading"
    #: The delta came out negative or zero — something else freed memory at
    #: the same time.
    IMPLAUSIBLE_DELTA = "implausible_delta"


@dataclass(frozen=True)
class MemoryEstimate:
    model_name: str
    #: None whenever the measurement was invalidated. The UI shows nothing.
    estimated_gb: float | None
    invalid: Invalid | None = None
    detail: str | None = None

    @property
    def usable(self) -> bool:
        return self.estimated_gb is not None and self.invalid is None


def _names(health: dict[str, Any] | None) -> set[str]:
    return {
        m["model_name"]
        for m in ((health or {}).get("all_models_loaded") or [])
        if isinstance(m, dict) and m.get("model_name")
    }


def _vram(stats: dict[str, Any] | None) -> float | None:
    value = (stats or {}).get("vram_gb")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def attribute(
    model_name: str,
    before_stats: dict[str, Any] | None,
    before_health: dict[str, Any] | None,
    after_stats: dict[str, Any] | None,
    after_health: dict[str, Any] | None,
) -> MemoryEstimate:
    """Attribute a VRAM delta to one model, or refuse.

    Refusing is the common case on a busy box, and that is correct.
    """
    before_vram, after_vram = _vram(before_stats), _vram(after_stats)
    if before_vram is None or after_vram is None:
        return MemoryEstimate(
            model_name, None, Invalid.NO_READING,
            "One of the VRAM readings was missing, so there is no delta to attribute.",
        )

    before_names, after_names = _names(before_health), _names(after_health)

    if model_name not in after_names:
        return MemoryEstimate(
            model_name, None, Invalid.TARGET_ABSENT,
            f"{model_name} is not loaded after the operation, so the delta is not its footprint.",
        )

    # The ONLY legitimate difference is the target appearing. Anything else —
    # an eviction the load triggered, another client's load, the nuclear
    # option — pollutes the delta.
    appeared = after_names - before_names
    vanished = before_names - after_names
    unexpected = (appeared - {model_name}) | vanished
    if unexpected:
        return MemoryEstimate(
            model_name, None, Invalid.COMPOSITION_CHANGED,
            f"The set of loaded models changed during the measurement ({', '.join(sorted(unexpected))} "
            f"appeared or was evicted), so the VRAM delta is not {model_name}'s alone. "
            f"Showing nothing rather than a number polluted by someone else's load.",
        )

    if model_name in before_names:
        return MemoryEstimate(
            model_name, None, Invalid.COMPOSITION_CHANGED,
            f"{model_name} was already loaded before the measurement, so the delta does not "
            f"represent loading it.",
        )

    delta = after_vram - before_vram
    if delta <= 0:
        return MemoryEstimate(
            model_name, None, Invalid.IMPLAUSIBLE_DELTA,
            f"VRAM did not increase ({before_vram:.1f} -> {after_vram:.1f} GB) even though "
            f"{model_name} loaded. Something else freed memory at the same moment; the delta "
            f"is meaningless.",
        )

    return MemoryEstimate(
        model_name, round(delta, 2), None,
        f"Estimated from the VRAM delta across the load ({before_vram:.1f} -> {after_vram:.1f} GB) "
        f"with no other model appearing or being evicted. An estimate, not a measurement: "
        f"Lemonade reports no per-model memory.",
    )
