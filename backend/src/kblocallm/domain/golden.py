"""Golden config: the operator-blessed baseline, and drift from it.

The ask (KAIROS/KODIAK, 2026-10-29): since the 2026.42.1 upgrade the local 26B
answers better, and the lever is a single load arg — `--reasoning off`. A reload
or a future Lemonade upgrade can silently drop it (or change a slot count, a
backend, the engine build) and nobody notices until answers regress. HELM cannot
PREVENT the box changing it — HELM does not own lemond's startup — but it can
hold a golden baseline and LOUDLY flag any drift of the running config from it.

Detect-and-alert ONLY. No auto re-apply: forcing the golden back is a
consequential write on inferred state, the same line the spec draws at
warm-on-startup, and it waits on the architect.

Pure and table-tested. The comparison tracks only the GOLDEN-SIGNIFICANT flags,
so Lemonade's injected companions (`--spec-type`, `--mmproj`, `--model-draft`)
never read as drift. `--reasoning` is flagged CRITICAL — it is the one that
changed the answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .llamacpp_args import find_flag

# The flags whose drift matters, with their aliases and a label. Sampling +
# slots + the reasoning switch + the load mode — the things that change what the
# model produces or how it is served. NOT an exhaustive arg diff: tracking a set
# is what keeps Lemonade's injected flags from reading as false drift.
_TRACKED_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reasoning", ("--reasoning",)),
    ("parallel", ("--parallel", "-np")),
    ("temp", ("--temp", "--temperature")),
    ("top_k", ("--top-k",)),
    ("top_p", ("--top-p",)),
    ("min_p", ("--min-p",)),
    ("repeat_penalty", ("--repeat-penalty",)),
    ("load_mode", ("--load-mode",)),
    ("slot_prompt_similarity", ("--slot-prompt-similarity",)),
)

#: The one that changed the answers. Drift here is loud.
_CRITICAL_KEYS = frozenset({"reasoning"})

_BUILD_RE = re.compile(r"llama-b(\d+)", re.IGNORECASE)


class GoldenState(str, Enum):
    MATCH = "match"
    DRIFT = "drift"
    #: Golden exists but the running config could not be read — never guessed as
    #: a match. The dashboard says "can't verify", not "all good".
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GoldenConfig:
    """The stored baseline. Captured from a model's current config at pin time."""
    model_id: str
    ctx_size: int | None = None
    llamacpp_backend: str | None = None
    backend_build: str | None = None
    llamacpp_args: str = ""
    note: str = ""
    captured_at: str | None = None


@dataclass(frozen=True)
class GoldenDriftItem:
    key: str
    golden: str | None
    running: str | None
    critical: bool = False


@dataclass(frozen=True)
class GoldenDrift:
    state: GoldenState
    items: list[GoldenDriftItem] = field(default_factory=list)
    #: True when any drifted item is the critical --reasoning one.
    has_critical: bool = False


def extract_backend_build(launch_command: Any) -> str | None:
    """The llama.cpp build from a /health launch_command (list argv or string),
    e.g. 'llama-b11394'. None when not present."""
    if isinstance(launch_command, list):
        text = " ".join(str(x) for x in launch_command)
    elif isinstance(launch_command, str):
        text = launch_command
    else:
        return None
    m = _BUILD_RE.search(text)
    return f"llama-b{m.group(1)}" if m else None


def _flag_value(args: str | None, aliases: tuple[str, ...]) -> str | None:
    """Last value of any alias, as a string. None if absent or value-less.
    Reuses the F4 parser so spelling (`-np`, `--parallel=8`) is handled once."""
    occ = find_flag(args, aliases)
    if not occ:
        return None
    return occ[-1].value


def capture_from_record(model_id: str, record: dict[str, Any], note: str = "") -> GoldenConfig:
    """Snapshot a model's CURRENT config into a golden. Reads recipe_options
    (effective) for args/ctx/backend and launch_command for the engine build —
    the tracked-flag comparison makes injected flags in `llamacpp_args`
    harmless, so we can store the effective string as-is."""
    opts = record.get("recipe_options") if isinstance(record.get("recipe_options"), dict) else {}
    ctx = opts.get("ctx_size")
    return GoldenConfig(
        model_id=model_id,
        ctx_size=ctx if isinstance(ctx, int) and not isinstance(ctx, bool) else None,
        llamacpp_backend=opts.get("llamacpp_backend"),
        backend_build=extract_backend_build(record.get("launch_command")),
        llamacpp_args=opts.get("llamacpp_args") or "",
        note=note,
    )


def compare_golden(golden: GoldenConfig, record: dict[str, Any]) -> GoldenDrift:
    """Compare a model's running config against its golden. The record is a
    /health all_models_loaded entry."""
    opts = record.get("recipe_options") if isinstance(record.get("recipe_options"), dict) else None
    running_build = extract_backend_build(record.get("launch_command"))

    # If we can read neither the running args nor the build, we cannot compare —
    # say UNKNOWN rather than claim a match we did not verify.
    if opts is None and running_build is None:
        return GoldenDrift(state=GoldenState.UNKNOWN)

    opts = opts or {}
    running_args = opts.get("llamacpp_args") or ""
    items: list[GoldenDriftItem] = []

    for key, aliases in _TRACKED_FLAGS:
        g = _flag_value(golden.llamacpp_args, aliases)
        if g is None:
            continue  # golden does not pin this flag; nothing to hold the line on
        r = _flag_value(running_args, aliases)
        if _norm(g) != _norm(r):
            items.append(GoldenDriftItem(
                key=key, golden=g, running=r, critical=key in _CRITICAL_KEYS))

    if golden.ctx_size is not None:
        r_ctx = opts.get("ctx_size")
        if not (isinstance(r_ctx, int) and not isinstance(r_ctx, bool)) or r_ctx != golden.ctx_size:
            items.append(GoldenDriftItem(
                key="ctx_size", golden=str(golden.ctx_size),
                running=str(r_ctx) if r_ctx is not None else None))

    if golden.llamacpp_backend:
        r_be = opts.get("llamacpp_backend")
        if _norm(r_be) != _norm(golden.llamacpp_backend):
            items.append(GoldenDriftItem(
                key="backend", golden=golden.llamacpp_backend, running=r_be))

    if golden.backend_build and running_build and golden.backend_build != running_build:
        # The engine itself changed — an upgrade swapped the build. Worth seeing,
        # but not the output-changing kind of drift, so not critical.
        items.append(GoldenDriftItem(
            key="backend_build", golden=golden.backend_build, running=running_build))

    if not items:
        return GoldenDrift(state=GoldenState.MATCH)
    return GoldenDrift(
        state=GoldenState.DRIFT, items=items,
        has_critical=any(i.critical for i in items))


def _norm(v: str | None) -> str | None:
    return v.strip().lower() if isinstance(v, str) else v
