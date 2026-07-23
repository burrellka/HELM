"""Planning a load/unload before it happens. Spec §5.3/§5.4, Phase 2.

**This module is where Phase 2's danger lives, which is why it is pure.**
Every rule below is one the operator would otherwise have to hold in their
head at the moment they are least likely to: mid-incident, about to click a
button that drops a model six agents are using.

The three that bite:

1. **The operator NEVER does the ctx multiplication.** They think in "8 slots
   of 64K"; Lemonade wants `ctx_size: 524288`. HELM computes it. That
   asymmetry is the entire reason the settings editor exists (§5.3) — the raw
   total is the number that misled everyone for months.
2. **NPU exclusivity is a hard router rule** (§3): loading a second NPU model
   silently unloads the first. Unattended, that is an agent losing its brain
   mid-conversation. HELM must say WHICH model dies BEFORE the click.
3. **Reserved args hard-error server-side** (§5.3), and they are how Lemonade
   auto-wires MTP, vision and device placement. Catch them here, in the same
   parser the slot math uses (F4), so the two can never disagree about what
   `-c` means.

Nothing here performs I/O. `plan_load` takes the operator's intent plus the
current fleet state and returns a body to POST and a list of issues. The
endpoint decides whether to send it; the tests decide whether it is right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .llamacpp_args import (
    MIN_P_FLAGS,
    REPEAT_PENALTY_FLAGS,
    RESERVED_FLAGS,
    TEMP_FLAGS,
    TOP_K_FLAGS,
    TOP_P_FLAGS,
    ParseStatus,
    find_flag,
    find_reserved,
    lookup_int,
)

#: Recipes that run on the NPU. Only one NPU backend may be active at a time
#: (spec §3) — a router-level rule, not a preference.
NPU_RECIPES: frozenset[str] = frozenset({"flm", "ryzenai-llm"})

#: Slots are a llamacpp concept (F3). Kept in sync with domain.concurrency by
#: importing the same idea, not by a second literal list.
from .concurrency import SLOTTED_RECIPES  # noqa: E402


class Level(str, Enum):
    #: Blocks the action. HELM will not send it.
    ERROR = "error"
    #: Proceeds, but the operator must see it first. NPU eviction lives here:
    #: it is a legitimate action with a consequence they cannot un-ring.
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class PlanIssue:
    level: Level
    code: str
    message: str


@dataclass
class LoadIntent:
    """What the operator asked for, in the units they think in."""

    model_name: str
    recipe: str | None = None
    #: llamacpp only. None = let Lemonade decide.
    slots: int | None = None
    #: Per-slot, NOT the total. HELM does the multiplication.
    ctx_per_slot: int | None = None
    llamacpp_backend: str | None = None
    pinned: bool = False
    #: Sampling params, from labeled fields. None = leave unset (model default).
    #: Composed into llamacpp_args by HELM; the operator never types the flags.
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    #: The raw-args escape hatch (§5.3). Validated against RESERVED_FLAGS.
    extra_args: str = ""
    #: --save-options: persist to recipe_options rather than just running with it.
    save_options: bool = False


@dataclass
class LoadPlan:
    body: dict[str, Any]
    issues: list[PlanIssue] = field(default_factory=list)
    #: Models this load will unload. Populated for NPU exclusivity.
    evicts: list[str] = field(default_factory=list)
    #: The total HELM computed, shown to the operator as a consequence rather
    #: than asked of them as an input.
    computed_ctx_size: int | None = None

    @property
    def blocked(self) -> bool:
        return any(i.level is Level.ERROR for i in self.issues)

    @property
    def errors(self) -> list[PlanIssue]:
        return [i for i in self.issues if i.level is Level.ERROR]


def _loaded_by_name(health: dict[str, Any] | None) -> dict[str, dict]:
    models = (health or {}).get("all_models_loaded")
    if not isinstance(models, list):
        return {}
    return {m["model_name"]: m for m in models if isinstance(m, dict) and m.get("model_name")}


def plan_load(
    intent: LoadIntent,
    health: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
) -> LoadPlan:
    """Turn operator intent into a validated request body.

    `catalog` is a model record from /api/v1/models keyed by id (for
    `max_context_window` and `recipe`). Both it and `health` are optional:
    without them HELM loses cross-checks, never correctness. It refuses to
    invent either.
    """
    issues: list[PlanIssue] = []
    catalog = catalog or {}
    recipe = intent.recipe or catalog.get("recipe")

    if not intent.model_name:
        issues.append(PlanIssue(Level.ERROR, "no_model", "No model name given."))
        return LoadPlan(body={}, issues=issues)

    # --- reserved args (§5.3). Same parser as the slot math (F4). ----------
    reserved = find_reserved(intent.extra_args)
    if reserved:
        issues.append(PlanIssue(
            Level.ERROR, "reserved_args",
            f"Lemonade will reject these arguments: {', '.join(reserved)}. They are how it "
            f"auto-wires context, MTP, vision and device placement, so it owns them. Set "
            f"context with the slots and context-per-slot fields instead of passing them raw.",
        ))

    # A raw --parallel would fight the slots field and silently win or lose
    # depending on Lemonade's arg order. Refuse rather than pick.
    raw_parallel = lookup_int(intent.extra_args, ("--parallel", "-np"))
    if raw_parallel.status is not ParseStatus.ABSENT and intent.slots is not None:
        issues.append(PlanIssue(
            Level.ERROR, "duplicate_parallel",
            "--parallel is set both in the slots field and in the raw arguments. "
            "Remove it from the raw arguments; the slots field is what HELM does the "
            "context maths from, and two sources would disagree silently.",
        ))

    # Same rule for the sampling fields: a value set both in its own field and
    # in the raw args has two sources that disagree by arg order. Refuse.
    _sampling = [
        ("--temp", intent.temperature, TEMP_FLAGS),
        ("--top-k", intent.top_k, TOP_K_FLAGS),
        ("--top-p", intent.top_p, TOP_P_FLAGS),
        ("--min-p", intent.min_p, MIN_P_FLAGS),
        ("--repeat-penalty", intent.repeat_penalty, REPEAT_PENALTY_FLAGS),
    ]
    dup_sampling = [
        label for label, value, aliases in _sampling
        if value is not None and find_flag(intent.extra_args, aliases)
    ]
    if dup_sampling:
        issues.append(PlanIssue(
            Level.ERROR, "duplicate_sampling",
            f"These sampling arguments are set both in their own field and in the raw arguments: "
            f"{', '.join(dup_sampling)}. Remove them from the raw arguments; the fields own them.",
        ))

    is_npu = recipe in NPU_RECIPES
    is_slotted = recipe in SLOTTED_RECIPES

    # --- slots are llamacpp-only (F3/Rule 2) ------------------------------
    if intent.slots is not None and not is_slotted:
        issues.append(PlanIssue(
            Level.ERROR, "slots_not_supported",
            f"{recipe or 'This recipe'} has no slot concept — "
            + ("the NPU processes one request at a time regardless of context. "
               if is_npu else "slots are a llamacpp feature. ")
            + "Leave slots empty.",
        ))

    slots = intent.slots
    if is_slotted and slots is not None and slots < 1:
        issues.append(PlanIssue(
            Level.ERROR, "bad_slots", f"Slots must be at least 1, not {slots}.",
        ))
        slots = None

    # --- the multiplication the operator must never do (Rule 1) -----------
    ctx_size: int | None = None
    if intent.ctx_per_slot is not None:
        if intent.ctx_per_slot < 1:
            issues.append(PlanIssue(
                Level.ERROR, "bad_ctx", f"Context per slot must be at least 1, not {intent.ctx_per_slot}.",
            ))
        else:
            ceiling = catalog.get("max_context_window")
            if isinstance(ceiling, int) and intent.ctx_per_slot > ceiling:
                issues.append(PlanIssue(
                    Level.ERROR, "ctx_over_ceiling",
                    f"Context per slot ({intent.ctx_per_slot:,}) exceeds this model's trained "
                    f"ceiling ({ceiling:,}). Lemonade will not give you more context than the "
                    f"model was built for.",
                ))
            else:
                # THE multiplication. 8 slots x 64K -> ctx_size 524288.
                effective_slots = slots if (is_slotted and slots) else 1
                ctx_size = intent.ctx_per_slot * effective_slots
                if effective_slots > 1:
                    issues.append(PlanIssue(
                        Level.INFO, "ctx_computed",
                        f"HELM will send ctx_size {ctx_size:,} — that is {intent.ctx_per_slot:,} "
                        f"per slot across {effective_slots} slots, because Lemonade's ctx_size is "
                        f"the total divided between them.",
                    ))

    # --- NPU exclusivity (Rule 7). The one that costs an agent its brain. --
    loaded = _loaded_by_name(health)
    if is_npu:
        incumbents = [
            name for name, m in loaded.items()
            if m.get("recipe") in NPU_RECIPES and name != intent.model_name
        ]
        for name in incumbents:
            issues.append(PlanIssue(
                Level.WARNING, "npu_exclusivity",
                f"Loading {intent.model_name} will UNLOAD {name}. Only one NPU model can be "
                f"active at a time — the router enforces this, and anything currently using "
                f"{name} will start failing.",
            ))
        plan_evicts = incumbents
    else:
        plan_evicts = []

    if intent.model_name in loaded:
        issues.append(PlanIssue(
            Level.WARNING, "already_loaded",
            f"{intent.model_name} is already loaded. Applying new options reloads it: it drops "
            f"for roughly 10-20 seconds and any in-flight requests fail.",
        ))

    # --- memory. We do NOT pretend to estimate. ---------------------------
    # §5.3 wants a live memory estimate before commit. HELM cannot honestly
    # give one: Lemonade exposes no per-model memory (§5.5), so the footprint
    # of a model that has never been loaded is genuinely unknown. Saying
    # "needs ~14GB" from a guess is exactly the confident-wrong number this
    # product exists to avoid. State the headroom, which IS known, and stop.
    issues.append(_headroom_note(health))

    body: dict[str, Any] = {"model_name": intent.model_name}
    options: dict[str, Any] = {}
    if ctx_size is not None:
        options["ctx_size"] = ctx_size
    if is_slotted:
        args = _compose_args(intent, slots)
        if args:
            options["llamacpp_args"] = args
        if intent.llamacpp_backend:
            options["llamacpp_backend"] = intent.llamacpp_backend
    if intent.pinned:
        options["pinned"] = True
    if options:
        body["recipe_options"] = options
    if recipe:
        body["recipe"] = recipe
    if intent.save_options:
        body["save_options"] = True

    return LoadPlan(
        body=body,
        issues=[i for i in issues if i is not None],
        evicts=plan_evicts,
        computed_ctx_size=ctx_size,
    )


def _compose_args(intent: LoadIntent, slots: int | None) -> str:
    """Build llamacpp_args from the structured fields plus the escape hatch.

    Slots are emitted as `--parallel N` (the long form) regardless of how the
    operator spelled it, so what we send is always the form the parser reads
    back unambiguously on the next poll.
    """
    parts: list[str] = []
    if slots is not None:
        parts.append(f"--parallel {slots}")
    # Sampling params from their labeled fields, in the canonical spelling the
    # parser reads back. `_fmt` drops a trailing .0 so a whole number does not
    # come back as "1.0" when the operator typed "1".
    for flag, value in (
        ("--temp", intent.temperature),
        ("--top-k", intent.top_k),
        ("--top-p", intent.top_p),
        ("--min-p", intent.min_p),
        ("--repeat-penalty", intent.repeat_penalty),
    ):
        if value is not None:
            parts.append(f"{flag} {_fmt(value)}")
    extra = (intent.extra_args or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _fmt(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _headroom_note(health: dict[str, Any] | None) -> PlanIssue:
    loaded = _loaded_by_name(health)
    return PlanIssue(
        Level.INFO, "no_memory_estimate",
        f"HELM cannot estimate this model's memory footprint: Lemonade reports no per-model "
        f"memory, only a total. {len(loaded)} model(s) are currently resident. If the pool is "
        f"tight, the router will evict the least recently used unpinned model to make room.",
    )


def plan_unload(model_name: str, health: dict[str, Any] | None = None) -> LoadPlan:
    """Unload. Short, but the warnings are the whole value.

    Unloading the pinned 26B is a legitimate operator action and also the single
    most disruptive thing HELM can do. The point is not to prevent it; it is to
    make sure nobody does it thinking it is reversible in under ten seconds.
    """
    issues: list[PlanIssue] = []
    loaded = _loaded_by_name(health)

    if not model_name:
        return LoadPlan(body={}, issues=[PlanIssue(Level.ERROR, "no_model", "No model name given.")])

    model = loaded.get(model_name)
    if model is None:
        issues.append(PlanIssue(
            Level.WARNING, "not_loaded",
            f"{model_name} is not in the loaded list, so there may be nothing to unload.",
        ))
    else:
        if model.get("pinned") is True:
            issues.append(PlanIssue(
                Level.WARNING, "unloading_pinned",
                f"{model_name} is PINNED, which means it was deliberately protected from "
                f"eviction. Unloading it is not undone by the pin: it will stay unloaded until "
                f"something loads it again.",
            ))
        issues.append(PlanIssue(
            Level.WARNING, "in_flight",
            f"Any request in flight against {model_name} will fail. Reloading it costs roughly "
            f"10-20 seconds, and it does NOT come back automatically after a lemond restart.",
        ))

    return LoadPlan(body={"model_name": model_name}, issues=issues)


__all__ = [
    "LoadIntent", "LoadPlan", "Level", "PlanIssue",
    "plan_load", "plan_unload", "NPU_RECIPES", "RESERVED_FLAGS",
]
