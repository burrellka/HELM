"""SAVED vs RUNNING. Spec §5.3.

The review called this "the best feature in the product", and the reason is
that it falls straight out of data already on the wire, and nothing else has
it. A model can be running with options that were never persisted:

    lemonade load X --parallel 8     # runs with 8 slots, saves NOTHING

There is currently no way to see that divergence. When lemond restarts, the
running configuration silently reverts to whatever was saved — which may be
nothing at all.

**This is not hypothetical. On the Lemonade host, right now:**
  - `Gemma-4-26B` SAVED `--parallel 8`, RUNNING `--parallel 8 --temp 1.0
    --top-k 64 --top-p 0.95`. The sampling settings are not persisted; a
    restart loses them and nobody would notice until output changed.
  - `Aevum-0.6B` SAVED `{}`, RUNNING a full context and sampling set. Its
    entire configuration is ephemeral.

**The false-positive trap.** `llamacpp_backend: rocm` appears in RUNNING and
not in SAVED for every model on the box, because Lemonade auto-detects it. A
naive field-by-field diff flags that as drift on every row, and a drift
indicator that is always on is one nobody reads — the same failure mode F8
identified for `watchdog_reset`. So `NOT_SAVED` is a distinct, quieter class
than `DRIFT`, and auto-detected fields are marked benign.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .llamacpp_args import ParseStatus, parallel_slots

#: Fields Lemonade fills in itself when absent. Their presence in RUNNING and
#: absence in SAVED is normal and must not read as an incident.
AUTO_DETECTED_FIELDS: frozenset[str] = frozenset({"llamacpp_backend"})


class DriftKind(str, Enum):
    SAME = "same"
    #: Both sides set it, and they disagree. The running model is not what the
    #: config says it is.
    DRIFT = "drift"
    #: Running has it; saved does not. A restart LOSES this.
    NOT_SAVED = "not_saved"
    #: Saved has it; running does not. The saved value is not in effect.
    NOT_RUNNING = "not_running"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FieldDrift:
    field: str
    kind: DriftKind
    saved: Any = None
    running: Any = None
    #: True when this difference is expected and not worth alarm.
    benign: bool = False
    note: str | None = None


@dataclass
class DriftReport:
    fields: list[FieldDrift]

    @property
    def has_drift(self) -> bool:
        """Only real disagreement counts. Auto-detected fields do not."""
        return any(f.kind is DriftKind.DRIFT and not f.benign for f in self.fields)

    @property
    def has_unpersisted(self) -> bool:
        """Running configuration a restart would silently discard."""
        return any(f.kind is DriftKind.NOT_SAVED and not f.benign for f in self.fields)

    @property
    def notable(self) -> list[FieldDrift]:
        return [f for f in self.fields if f.kind is not DriftKind.SAME and not f.benign]


def _slots(options: dict[str, Any] | None) -> int | None:
    """Slot count from an options blob, via the shared parser (F4)."""
    parsed = parallel_slots((options or {}).get("llamacpp_args"))
    if parsed.status is ParseStatus.ABSENT:
        return 1  # llama.cpp's documented default
    return parsed.value if parsed.ok else None


def _compare(field: str, saved: Any, running: Any, note: str | None = None) -> FieldDrift:
    benign = field in AUTO_DETECTED_FIELDS
    if saved is None and running is None:
        kind = DriftKind.SAME
    elif saved is None:
        kind = DriftKind.NOT_SAVED
        note = note or (
            f"Lemonade auto-detects {field}; it not being saved is normal."
            if benign else
            f"The model is running with this, but it is NOT saved — a lemond restart "
            f"will silently drop it."
        )
    elif running is None:
        kind = DriftKind.NOT_RUNNING
        note = note or "Saved, but not in effect on the running model."
    elif saved == running:
        kind = DriftKind.SAME
    else:
        kind = DriftKind.DRIFT
        note = note or "The running model does not match what is saved."
    return FieldDrift(field=field, kind=kind, saved=saved, running=running,
                      benign=benign, note=note if kind is not DriftKind.SAME else None)


def compare_options(
    saved: dict[str, Any] | None,
    running: dict[str, Any] | None,
) -> DriftReport:
    """Diff a catalog `recipe_options` against a health `recipe_options`.

    `saved` comes from /api/v1/models (what persists), `running` from
    /api/v1/health (what is actually in memory). Both may be empty; an empty
    SAVED is itself the finding (Aevum on the live box).
    """
    saved = saved or {}
    running = running or {}

    fields = [
        _compare("slots", _slots(saved) if saved else None, _slots(running) if running else None),
        _compare("ctx_size", saved.get("ctx_size"), running.get("ctx_size")),
        _compare("llamacpp_backend", saved.get("llamacpp_backend"), running.get("llamacpp_backend")),
        _compare("pinned", saved.get("pinned"), running.get("pinned")),
        _compare("llamacpp_args", saved.get("llamacpp_args"), running.get("llamacpp_args")),
    ]
    return DriftReport(fields=fields)
