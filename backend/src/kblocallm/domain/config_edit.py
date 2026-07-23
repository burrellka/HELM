"""The guarded config-edit pipeline. Spec §5.3b/§5.3c, 14c Part 2 §2.3.

This is Kevin's own SOP as code: read current, validate before send, diff,
back up, confirm with the blast radius named, apply, one-click restore. The
raw editor is the power-user door; Export/Import is the same pipeline with a
file on the end. Reset-to-auto is the same pipeline writing empty overrides.

Everything in this module is a PURE function over text and dicts. No I/O, no
upstream calls. The validation is the safety rail, so it has to be
table-testable: a raw edit that flips `enable_dgpu_gtt` to false must fight the
operator on the way out, and the only way to trust that it always does is to
test it in isolation.

**Two levels, an order of magnitude apart in blast radius (14c §2.3):**
  - Level A — per-model `recipe_options`. Applying reloads ONE model.
  - Level B — global `config.json`. Holds `enable_dgpu_gtt` (the 80GB saga
    switch), ports, bindings. Applying is assumed to need a FULL server
    restart that drops every model, until the box says otherwise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .llamacpp_args import find_reserved, parallel_slots, ParseStatus

# ---------------------------------------------------------------------------
# Level A: the keys Lemonade's per-model recipe_options is known to accept.
# Unknown keys are WARNED, not blocked — this is an escape hatch, and Lemonade
# may accept recipe-specific keys we have not catalogued. But a typo like
# "ctx_sixe" should be visible, not silently ignored by the server.
KNOWN_RECIPE_KEYS: frozenset[str] = frozenset({
    "ctx_size", "llamacpp_args", "llamacpp_backend", "pinned",
})

# ---------------------------------------------------------------------------
# Level B: keys whose change is high-consequence. Editing any of these demands
# an explicit second confirmation. `enable_dgpu_gtt` is THE one — it is the
# 80GB saga switch. Ports and bindings can strand the whole fleet. Auth keys
# can lock everyone out or open the door.
DANGEROUS_GLOBAL_KEYS: frozenset[str] = frozenset({
    "enable_dgpu_gtt", "host", "port", "extra_models_dir", "models_dir",
})

#: Substrings that mark a key as secret/auth-bearing wherever it appears.
_SECRET_MARKERS: tuple[str, ...] = ("key", "token", "secret", "auth", "password")


class Level(str, Enum):
    ERROR = "error"      # blocks the apply
    DANGER = "danger"    # allowed, but needs an explicit extra confirmation
    WARNING = "warning"  # proceeds, operator should see it
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    level: Level
    code: str
    message: str
    #: The key this issue is about, when it is about one. Lets the UI point.
    key: str | None = None


@dataclass(frozen=True)
class FieldChange:
    key: str
    old: Any = None
    new: Any = None
    #: "added" | "removed" | "changed"
    kind: str = "changed"


@dataclass
class ConfigValidation:
    issues: list[Issue] = field(default_factory=list)
    changes: list[FieldChange] = field(default_factory=list)
    #: The parsed new config, when it parsed. None on a parse error.
    parsed: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return any(i.level is Level.ERROR for i in self.issues)

    @property
    def needs_danger_ack(self) -> bool:
        return any(i.level is Level.DANGER for i in self.issues)

    @property
    def dangerous_keys(self) -> list[str]:
        return [i.key for i in self.issues if i.level is Level.DANGER and i.key]


def parse_config(text: str) -> tuple[dict[str, Any] | None, Issue | None]:
    """Parse the edited text as JSON. A broken config.json is refused OUTRIGHT.

    This is the first gate and the most important one for Level B: the server
    will not restart on a config.json that does not parse, so a trailing comma
    could brick it. HELM refuses to send text it cannot parse, every time.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        return None, Issue(
            Level.ERROR, "parse_error",
            f"This is not valid JSON, so HELM will not send it: {e}. A malformed config is "
            f"how you brick a server that will not restart.",
        )
    if not isinstance(parsed, dict):
        return None, Issue(
            Level.ERROR, "not_an_object",
            f"Config must be a JSON object, got {type(parsed).__name__}.",
        )
    return parsed, None


def diff_config(old: dict[str, Any], new: dict[str, Any]) -> list[FieldChange]:
    """Old vs new, key by key. Never a blind overwrite (14c §2.3 step 4)."""
    changes: list[FieldChange] = []
    for key in sorted(set(old) | set(new)):
        in_old, in_new = key in old, key in new
        if in_old and not in_new:
            changes.append(FieldChange(key, old=old[key], kind="removed"))
        elif in_new and not in_old:
            changes.append(FieldChange(key, new=new[key], kind="added"))
        elif old[key] != new[key]:
            changes.append(FieldChange(key, old=old[key], new=new[key], kind="changed"))
    return changes


def validate_recipe_options(
    text: str,
    current: dict[str, Any] | None = None,
    recipe: str | None = None,
    max_context_window: int | None = None,
) -> ConfigValidation:
    """Level A: validate a raw per-model recipe_options edit.

    Runs the SAME reserved-arg check the structured form uses (the shared F4
    parser), so a raw edit cannot smuggle in `--ctx-size` where the form would
    have caught it. Raw editing is more powerful, not more correct — this is
    what keeps it from being a footgun.
    """
    parsed, err = parse_config(text)
    if err is not None:
        return ConfigValidation(issues=[err])

    issues: list[Issue] = []

    args = parsed.get("llamacpp_args")
    if args is not None and not isinstance(args, str):
        issues.append(Issue(Level.ERROR, "args_type",
                            "llamacpp_args must be a string.", key="llamacpp_args"))
    elif isinstance(args, str):
        reserved = find_reserved(args)
        if reserved:
            issues.append(Issue(
                Level.ERROR, "reserved_args",
                f"Lemonade will reject these reserved arguments: {', '.join(reserved)}. They are "
                f"how it auto-wires context and device placement; it owns them.",
                key="llamacpp_args",
            ))

    ctx = parsed.get("ctx_size")
    if ctx is not None:
        if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1:
            issues.append(Issue(Level.ERROR, "ctx_type",
                                f"ctx_size must be a positive integer, got {ctx!r}.", key="ctx_size"))
        elif max_context_window and isinstance(args, str):
            parsed_slots = parallel_slots(args)
            slots = 1 if parsed_slots.status is ParseStatus.ABSENT else parsed_slots.value
            if slots:
                per_slot = ctx // slots
                if per_slot > max_context_window:
                    issues.append(Issue(
                        Level.ERROR, "ctx_over_ceiling",
                        f"ctx_size {ctx:,} across {slots} slot(s) is {per_slot:,} per slot, over "
                        f"this model's trained ceiling of {max_context_window:,}.",
                        key="ctx_size",
                    ))

    for key in parsed:
        if key not in KNOWN_RECIPE_KEYS:
            issues.append(Issue(
                Level.WARNING, "unknown_key",
                f"'{key}' is not a recipe option HELM knows. Lemonade may accept it, or it may "
                f"be a typo it silently ignores. Double-check it.",
                key=key,
            ))

    return ConfigValidation(
        issues=issues, parsed=parsed,
        changes=diff_config(current or {}, parsed),
    )


def validate_global_config(
    text: str,
    current: dict[str, Any] | None = None,
) -> ConfigValidation:
    """Level B: validate a raw global config.json edit. High blast radius.

    Dangerous keys get a DANGER issue that the UI must surface as an explicit
    second confirmation. `enable_dgpu_gtt` flipping to false is called out by
    name — it is the exact edit that caused the 80GB saga, and a raw editor
    that let it sail through would be the foot-gun this pipeline exists to
    prevent.
    """
    parsed, err = parse_config(text)
    if err is not None:
        return ConfigValidation(issues=[err])

    issues: list[Issue] = []
    current = current or {}
    changes = diff_config(current, parsed)
    changed_keys = {c.key for c in changes}

    for key in sorted(changed_keys):
        lowered = key.lower()
        is_secret = any(marker in lowered for marker in _SECRET_MARKERS)
        if key in DANGEROUS_GLOBAL_KEYS or is_secret:
            issues.append(Issue(
                Level.DANGER, "dangerous_key",
                _danger_message(key, current.get(key), parsed.get(key)),
                key=key,
            ))

    # The saga's signature edit, called out specifically even beyond the
    # generic dangerous-key flag.
    if parsed.get("enable_dgpu_gtt") is False and current.get("enable_dgpu_gtt") is not False:
        issues.append(Issue(
            Level.DANGER, "gtt_disable",
            "You are turning enable_dgpu_gtt OFF. This makes Lemonade see only the small BIOS "
            "carve-out instead of the full unified pool — it is the exact switch behind the 80GB "
            "saga. Do this only if you are certain.",
            key="enable_dgpu_gtt",
        ))

    return ConfigValidation(issues=issues, parsed=parsed, changes=changes)


def _danger_message(key: str, old: Any, new: Any) -> str:
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        # Never echo the value of a secret in a message.
        return (
            f"'{key}' looks like a credential. Changing it can lock the fleet out or open it up. "
            f"Confirm explicitly."
        )
    if key in ("host", "port"):
        return (
            f"Changing '{key}' ({old!r} -> {new!r}) moves where Lemonade listens. Get it wrong "
            f"and every client, HELM included, loses the server until it is fixed on the box."
        )
    return f"'{key}' is a high-consequence setting ({old!r} -> {new!r}). Confirm explicitly."
