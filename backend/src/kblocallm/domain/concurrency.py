"""Concurrency + per-slot context. Spec Rule 1, review F3, handoff §3.

Two rules meet here, and both exist because the CLI dashboard this replaces
got them wrong:

**F3 — slots are an ALLOWLIST, not a denylist.** The old shape was
`if flm: SERIAL else: slots`, which fabricates a slot count for every recipe
that isn't llamacpp or flm. The live box carries ten recipes (whispercpp,
sd-cpp, kokoro, moonshine, vllm, ...) and §5B.4 actively pushes installing
more, so the `else` branch was a loaded gun. Here: `llamacpp` derives slots.
Nothing else does. Ever.

**Rule 1 — `ctx_size` is a TOTAL, divided across slots.** The 26B on the live
box reads `ctx_size: 524288` with `--parallel 8`. That is 8 slots of 65,536,
NOT 8 slots of 524,288. Showing the raw number is actively misleading, so
this module only ever emits the derived per-slot value — and emits `unknown`
rather than a computed one whenever the parse it depends on didn't hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .llamacpp_args import ParseStatus, parallel_slots

#: The only recipe with parallel-slot semantics. Live-validated: system-info's
#: `recipes` block gives llamacpp `slot_policy: standard` + `uses_ctx_size:
#: true`, and it is the only one whose per-model args carry `--parallel`.
#: Adding to this set means teaching the parser that recipe's arg syntax too —
#: vllm has slots but they are NOT spelled in `llamacpp_args`, so it does not
#: belong here.
SLOTTED_RECIPES: frozenset[str] = frozenset({"llamacpp"})

#: The NPU runtime. Serial by router construction (spec §3: one NPU backend
#: active at a time, one request at a time regardless of ctx_size). This is a
#: documented domain fact, which is why FLM gets a stronger claim than the
#: recipes we simply have no slot knowledge about.
SERIAL_RECIPES: frozenset[str] = frozenset({"flm"})


class ConcurrencyKind(str, Enum):
    SLOTS = "slots"                    # llamacpp, parse held
    SERIAL = "serial"                  # flm: one request at a time
    NOT_APPLICABLE = "not_applicable"  # no slot concept for this recipe
    UNKNOWN = "unknown"                # we could not read it. Say so.


@dataclass(frozen=True)
class Concurrency:
    kind: ConcurrencyKind
    #: Tested display string. Computed here, not in the frontend, so the
    #: string the operator reads is the string the test asserts.
    display: str
    slots: int | None = None
    per_slot_ctx: int | None = None
    #: True when `--parallel` was absent and llama.cpp's default of 1 applies.
    #: The operator did not type this; the display says so.
    slots_defaulted: bool = False
    #: Verbatim `recipes[recipe].slot_policy` from system-info when known.
    #: Enrichment only — it never drives the slot math (F3).
    slot_policy: str | None = None
    #: Why it's unknown / what we did read. Actionable detail beats a bare
    #: "unknown".
    note: str | None = None


def format_ctx(ctx: int) -> str:
    """65536 -> '64K'. Non-round numbers keep their digits rather than being
    rounded into a tidier lie."""
    if ctx >= 1024 and ctx % 1024 == 0:
        return f"{ctx // 1024}K"
    return f"{ctx:,}"


def _ctx_size(recipe_options: dict[str, Any]) -> int | None:
    ctx = recipe_options.get("ctx_size")
    if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx <= 0:
        return None
    return ctx


def derive_concurrency(
    model: dict[str, Any],
    recipes: dict[str, Any] | None = None,
) -> Concurrency:
    """The one place slots and per-slot context are decided.

    `recipes` is system-info's `recipes` block, optional: it only supplies
    `slot_policy` for display and `uses_ctx_size` to decide whether a context
    is meaningful for this recipe at all. Passing None costs enrichment, never
    correctness.
    """
    recipe = model.get("recipe") or ""
    options = model.get("recipe_options") or {}
    recipe_meta = (recipes or {}).get(recipe) or {}
    slot_policy = recipe_meta.get("slot_policy")
    # Absent metadata: assume a context IS meaningful, so we show what we have
    # rather than hiding a real number behind an assumption.
    uses_ctx = recipe_meta.get("uses_ctx_size", True)
    ctx = _ctx_size(options) if uses_ctx else None

    # ---- the allowlist. Everything below turns on this line. --------------
    if recipe not in SLOTTED_RECIPES:
        if recipe in SERIAL_RECIPES:
            display = f"SERIAL @ {format_ctx(ctx)}" if ctx else "SERIAL"
            return Concurrency(
                kind=ConcurrencyKind.SERIAL,
                display=display,
                per_slot_ctx=ctx,
                slot_policy=slot_policy,
                note="NPU processes one request at a time regardless of context.",
            )

        if not recipe:
            return Concurrency(
                kind=ConcurrencyKind.UNKNOWN,
                display="unknown",
                note="The health record carries no recipe, so we cannot say.",
            )

        # A recipe we have no slot knowledge about (whispercpp, sd-cpp, ...).
        # "n/a" is the honest answer: not a fabricated slot count, and not a
        # SERIAL claim we never verified for this backend.
        return Concurrency(
            kind=ConcurrencyKind.NOT_APPLICABLE,
            display="n/a",
            per_slot_ctx=ctx,
            slot_policy=slot_policy,
            note=f"Slots are a llamacpp concept; {recipe} has no slot count.",
        )

    # ---- llamacpp: derive. -----------------------------------------------
    parsed = parallel_slots(options.get("llamacpp_args"))

    if parsed.status is ParseStatus.MALFORMED:
        # F4's contract, and the reason this module exists. A malformed parse
        # NEVER becomes `default 1`, because that renders ctx_size as the
        # per-slot value — 8x the truth, stated confidently.
        return Concurrency(
            kind=ConcurrencyKind.UNKNOWN,
            display="unknown",
            slot_policy=slot_policy,
            note=(
                f"Could not read the slot count from llamacpp_args ({parsed.raw!r}), "
                f"so the per-slot context cannot be derived."
            ),
        )

    defaulted = parsed.status is ParseStatus.ABSENT
    slots = 1 if defaulted else parsed.value
    assert slots is not None and slots >= 1  # parallel_slots guarantees it

    if ctx is None:
        note = (
            "No usable ctx_size on the model record."
            if uses_ctx else
            f"{recipe} does not use ctx_size."
        )
        return Concurrency(
            kind=ConcurrencyKind.SLOTS,
            display=f"{_slots_label(slots, defaulted)} × unknown ctx",
            slots=slots,
            slots_defaulted=defaulted,
            slot_policy=slot_policy,
            note=note,
        )

    per_slot = ctx // slots
    return Concurrency(
        kind=ConcurrencyKind.SLOTS,
        display=f"{_slots_label(slots, defaulted)} × {format_ctx(per_slot)}",
        slots=slots,
        per_slot_ctx=per_slot,
        slots_defaulted=defaulted,
        slot_policy=slot_policy,
        note=(
            f"ctx_size {ctx:,} is the TOTAL across {slots} slots."
            if slots > 1 else None
        ),
    )


def _slots_label(slots: int, defaulted: bool) -> str:
    """'8 slots' vs '1 slot (default)'.

    The parenthetical is not decoration: an absent `--parallel` means we are
    quoting llama.cpp's default back at the operator, not reading a setting
    they made. Those are different claims and the column should say which.
    """
    noun = "slot" if slots == 1 else "slots"
    return f"{slots} {noun} (default)" if defaulted else f"{slots} {noun}"
