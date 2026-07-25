"""Running-context drift: does the context a llamacpp model is ACTUALLY running
match what HELM computes from its saved options? (14c §1.12, a §6 correctness fix.)

HELM shows per-slot context as `ctx_size / slots`, computed from recipe_options.
That is what the operator SET. What llama.cpp actually ALLOCATED can differ — it
can cap context below the request under memory pressure — and a control plane that
shows the configured number as if it were the running truth is telling the exact
kind of confident lie §6 exists to prevent.

llama.cpp exposes the running per-slot `n_ctx` via `/slots`, which Lemonade
proxies (reachable, portable). But there is a hard limit, verified against the
box: **Lemonade's `/api/v1/slots` is NOT per-model** — it returns one llama.cpp
instance's slots, with no model label, and the `?model` param is ignored. So we
cannot ask "what is Aevum's n_ctx"; we can only read whatever instance `/slots`
resolves to.

The honest way through, and the whole point of this module: attribute `/slots` to
a model by its **slot count** (`--parallel`), which is independent of the context
we are checking — then compare that model's computed per-slot context against the
reported `n_ctx`. If the slot count does not uniquely identify one loaded llamacpp
model, we CANNOT know whose slots these are, so drift is **unknown** — never
"no drift". Same tri-state discipline as the F9 label join: an answer we can stand
behind, or a visible unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .concurrency import SLOTTED_RECIPES
from .llamacpp_args import ParseStatus, parallel_slots


class DriftState(str, Enum):
    MATCH = "match"        # attributed, and computed per-slot == running n_ctx
    DRIFT = "drift"        # attributed, and they DIFFER — the running value is truth
    UNKNOWN = "unknown"    # cannot attribute /slots to this model, or no data


@dataclass(frozen=True)
class CtxDrift:
    state: DriftState
    #: What HELM computes/displays: ctx_size / slots. None when unknowable.
    computed_per_slot: int | None = None
    #: What llama.cpp reports running, when we can attribute it to THIS model.
    running_n_ctx: int | None = None
    #: Human-readable why — especially for UNKNOWN, which must say what it could
    #: not determine rather than read as reassurance.
    detail: str = ""


def _target_slots_and_ctx(record: dict[str, Any]) -> tuple[int | None, int | None]:
    """(slot count, per-slot ctx) HELM computes for a model, or (None, ...)."""
    opts = record.get("recipe_options") or {}
    args = opts.get("llamacpp_args")
    parsed = parallel_slots(args)
    if parsed.status is ParseStatus.ABSENT:
        slots = 1                       # llama.cpp's documented default
    elif parsed.ok and parsed.value:
        slots = parsed.value
    else:
        slots = None                    # malformed --parallel: we don't guess
    ctx = opts.get("ctx_size")
    ctx = ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else None
    per_slot = (ctx // slots) if (slots and ctx) else None
    return slots, per_slot


def _slots_ctx(slots_payload: Any) -> tuple[int | None, int | None]:
    """(slot count, the single n_ctx common to all slots) from a /slots payload.

    A payload whose slots disagree on n_ctx is not something to average — it is
    something we cannot read cleanly, so n_ctx comes back None (-> unknown)."""
    if not isinstance(slots_payload, list) or not slots_payload:
        return None, None
    n_ctxs = {s.get("n_ctx") for s in slots_payload if isinstance(s, dict)}
    n_ctxs.discard(None)
    n_ctx = next(iter(n_ctxs)) if len(n_ctxs) == 1 else None
    return len(slots_payload), n_ctx


def check_ctx_drift(
    record: dict[str, Any],
    loaded_llamacpp: list[dict[str, Any]],
    slots_payload: Any,
) -> CtxDrift:
    """Compute the running-context drift for ONE model.

    `record` is the target model's catalog/health record; `loaded_llamacpp` is
    every currently-loaded llamacpp model (to detect the ambiguity that forces
    unknown); `slots_payload` is Lemonade's `/api/v1/slots` (or None if the call
    failed — a failure is unknown, never no-drift).
    """
    recipe = record.get("recipe")
    if recipe not in SLOTTED_RECIPES:
        return CtxDrift(DriftState.UNKNOWN,
                        detail=f"{recipe or 'this recipe'} has no llama.cpp slots to read "
                               f"(running context is not observable this way).")
    if not record.get("loaded"):
        return CtxDrift(DriftState.UNKNOWN,
                        detail="Model is not loaded, so it has no running slots to read.")

    target_slots, computed = _target_slots_and_ctx(record)
    if target_slots is None or computed is None:
        return CtxDrift(DriftState.UNKNOWN, computed_per_slot=computed,
                        detail="HELM cannot compute this model's per-slot context "
                               "(unreadable --parallel or missing ctx_size), so there is "
                               "nothing to compare against.")

    slot_count, n_ctx = _slots_ctx(slots_payload)
    if slot_count is None or n_ctx is None:
        return CtxDrift(DriftState.UNKNOWN, computed_per_slot=computed,
                        detail="Live slot state was not available to read.")

    # Attribute /slots to a model by its slot count — the one signal that does
    # not depend on the context we're checking. If this model's slot count does
    # not match, /slots is reporting a DIFFERENT model. If more than one loaded
    # llamacpp model shares this slot count, /slots is ambiguous (it carries no
    # model label). Either way: unknown, never a guess.
    same_count = [
        m for m in loaded_llamacpp
        if _target_slots_and_ctx(m)[0] == slot_count
    ]
    if target_slots != slot_count or len(same_count) != 1:
        return CtxDrift(
            DriftState.UNKNOWN, computed_per_slot=computed, running_n_ctx=None,
            detail="Lemonade's /slots is not per-model and could not be attributed to "
                   "this model (another loaded model shares its slot count, or the live "
                   "slots belong to a different model). The running context is unknown.")

    if n_ctx == computed:
        return CtxDrift(DriftState.MATCH, computed_per_slot=computed, running_n_ctx=n_ctx,
                        detail="The running per-slot context matches what HELM computed.")
    return CtxDrift(
        DriftState.DRIFT, computed_per_slot=computed, running_n_ctx=n_ctx,
        detail=f"llama.cpp is running {n_ctx:,} tokens per slot, but HELM computed "
               f"{computed:,} from the saved options. The running value is the truth; "
               f"the model may have capped context under memory pressure.")
