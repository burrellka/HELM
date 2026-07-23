"""The model library. Spec §5.2, §4.3, §5.3a. Phase 3.

**Rule 10 is the reason this file is careful.** On the NPU, tool calling is
compiled per-model into the `.q4nx` kernel. FLM **silently strips the `tools`
array** for models that lack it — the model never sees the tools and
hallucinates an answer instead. The `tool-calling` label is the ONLY reliable
indicator. Not the name, not the size, not the family.

That is not theoretical on this box. Of the downloaded FLM models:
    gemma4-it-e2b-FLM     tool-calling  ✓
    qwen3.5-2b-FLM        tool-calling  ✓
    llama3.2-1b-FLM       labels: []    ← wires an agent to silent failure
    phi4-mini-it-4b-FLM   labels: []    ←
    qwen2.5-it-3b-FLM     labels: []    ←
    lfm2.5-it-1.2b-FLM    labels: []    ←

**And that is exactly why F9's miss behaviour matters.** A model showing no
`tool-calling` badge because the JOIN FAILED looks identical to one that
genuinely lacks the capability. Absence of evidence is not evidence of
absence. So the join result is tri-state — MATCHED / no labels / UNKNOWN — and
an unmatched model renders "capabilities unknown", never an empty badge row.

The join key is `health.all_models_loaded[].model_name == models.data[].id`,
live-confirmed 2/2 on the box (14a RESULTS §1) and re-confirmed 3/3 while
building Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .concurrency import SLOTTED_RECIPES  # F3's allowlist. One definition, imported.
from .drift import DriftReport, compare_options
from .llamacpp_args import (
    MIN_P_FLAGS,
    PARALLEL_FLAGS,
    REPEAT_PENALTY_FLAGS,
    SAMPLING_FLAGS,
    TEMP_FLAGS,
    TOP_K_FLAGS,
    TOP_P_FLAGS,
    ParseStatus,
    lookup_float,
    lookup_int,
    parallel_slots,
    strip_flag,
)

#: Labels Lemonade publishes that describe a real capability. `custom` and
#: `hot` are provenance/curation markers, not capabilities, and badging them
#: beside `tool-calling` would dilute the one badge that prevents silent
#: failure.
CAPABILITY_LABELS: frozenset[str] = frozenset({
    "tool-calling", "vision", "audio", "reasoning", "embeddings",
    "mtp", "coding", "reranking", "chat-transcription",
})

#: Recipes where a missing `tool-calling` label means tools are silently
#: STRIPPED rather than merely unsupported. This is the FLM/NPU behaviour that
#: makes Rule 10 a correctness rule instead of a UI preference.
TOOL_STRIPPING_RECIPES: frozenset[str] = frozenset({"flm"})

#: Models that are not chat models at all. Tool calling is not a thing you
#: would ever ask of them, so warning that they "silently strip tools" is
#: technically true and operationally noise — nobody wires an agent to an
#: embedding model expecting a tool call. A warning that fires where it cannot
#: matter is how the warning that DOES matter gets ignored (the same failure
#: F8 identified for a permanently-lit watchdog flag).
NON_CHAT_LABELS: frozenset[str] = frozenset({"embeddings", "reranking"})


class LabelJoin(str, Enum):
    #: Joined, and the model publishes labels.
    MATCHED = "matched"
    #: Joined, and the model genuinely publishes none. A real, load-bearing
    #: answer for an FLM model: it will strip tools.
    NO_LABELS = "no_labels"
    #: The join FAILED. We know nothing. Must never render as "no capabilities".
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelSettings:
    """One side (saved or running) of a model's configuration, in the units the
    OPERATOR thinks in — per-slot context, not Lemonade's total.

    Derived here, not in the frontend. The division is Rule 1 read backwards
    (`per_slot = ctx_size / slots`), and the args stripping needs the F4
    parser; doing either in TypeScript would be a second implementation of both
    and would drift from this one the first time somebody typed `-np`.
    """

    slots: int | None = None
    #: PER SLOT. This is what the editor's field shows.
    ctx_per_slot: int | None = None
    #: The raw total, for reference. Never an input.
    ctx_size: int | None = None
    backend: str | None = None
    #: Sampling params, lifted out of llamacpp_args into their own fields so the
    #: editor shows them as labeled inputs, not raw text. None = not set (use the
    #: model default). All from the one shared parser.
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    #: llamacpp_args with --parallel AND the sampling flags removed: each of
    #: those owns its own field, and the planner refuses a value with two
    #: sources. What remains is the genuine escape hatch.
    extra_args: str = ""


def _settings_from(options: dict[str, Any] | None, recipe: str | None) -> ModelSettings:
    options = options or {}
    args = options.get("llamacpp_args")
    slotted = recipe in SLOTTED_RECIPES

    parsed = parallel_slots(args) if slotted else None
    if parsed is None:
        slots = None
    elif parsed.status is ParseStatus.ABSENT:
        slots = 1  # llama.cpp's documented default
    else:
        slots = parsed.value if parsed.ok else None

    ctx = options.get("ctx_size")
    ctx = ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else None

    per_slot: int | None = None
    if ctx is not None:
        # Refuse to divide by a slot count we could not read. Guessing 1 here
        # would render the TOTAL as the per-slot value — the 8x lie.
        per_slot = ctx // slots if (slotted and slots) else (None if slotted else ctx)

    # Sampling params, lifted into their own fields (only meaningful for the
    # slotted/llamacpp recipe, where they ride in llamacpp_args).
    top_k_flag = lookup_int(args, TOP_K_FLAGS) if slotted else None
    settings_extra = (
        strip_flag(args, PARALLEL_FLAGS + SAMPLING_FLAGS) if slotted else (args or "")
    )
    return ModelSettings(
        slots=slots,
        ctx_per_slot=per_slot,
        ctx_size=ctx,
        backend=options.get("llamacpp_backend"),
        temperature=lookup_float(args, TEMP_FLAGS) if slotted else None,
        top_k=top_k_flag.value if (top_k_flag and top_k_flag.ok) else None,
        top_p=lookup_float(args, TOP_P_FLAGS) if slotted else None,
        min_p=lookup_float(args, MIN_P_FLAGS) if slotted else None,
        repeat_penalty=lookup_float(args, REPEAT_PENALTY_FLAGS) if slotted else None,
        extra_args=settings_extra,
    )


@dataclass
class LibraryEntry:
    model_id: str
    recipe: str | None = None
    checkpoint: str | None = None
    downloaded: bool = False
    suggested: bool = False
    update_available: bool = False
    size: int | None = None
    max_context_window: int | None = None

    #: Lemonade's own labels, filtered to real capabilities.
    labels: list[str] = field(default_factory=list)
    #: Everything Lemonade published, unfiltered — provenance included.
    raw_labels: list[str] = field(default_factory=list)
    label_join: LabelJoin = LabelJoin.UNKNOWN

    loaded: bool = False
    pinned: bool = False
    device: str | None = None

    #: §5.3a — HELM-local, no upstream source of truth.
    description: str = ""
    favorite: bool = False
    archived: bool = False
    tags: list[str] = field(default_factory=list)

    drift: DriftReport | None = None
    #: What is persisted. Present for any catalog entry with saved options.
    saved_settings: ModelSettings | None = None
    #: What is in memory right now. Only for loaded models.
    running_settings: ModelSettings | None = None

    @property
    def strips_tools_silently(self) -> bool:
        """True when wiring an agent to this model would fail QUIETLY.

        Only claimed when the join actually succeeded. If we could not
        establish the labels, we do not know — and saying "this strips tools"
        on a failed join would be a different confident lie in the opposite
        direction.
        """
        if self.label_join is LabelJoin.UNKNOWN:
            return False
        if self.recipe not in TOOL_STRIPPING_RECIPES:
            return False
        if NON_CHAT_LABELS & set(self.labels):
            return False
        return "tool-calling" not in self.labels


def build_library(
    catalog: dict[str, Any] | None,
    health: dict[str, Any] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> list[LibraryEntry]:
    """Join the catalog, the loaded state, and HELM's own metadata.

    `catalog` is /api/v1/models?show_all=true; `health` is /api/v1/health;
    `metadata` is HELM's SQLite, keyed by model id.
    """
    metadata = metadata or {}
    data = (catalog or {}).get("data")
    data = data if isinstance(data, list) else []

    loaded_by_name: dict[str, dict] = {}
    for model in (health or {}).get("all_models_loaded") or []:
        if isinstance(model, dict) and model.get("model_name"):
            loaded_by_name[model["model_name"]] = model

    entries: list[LibraryEntry] = []
    for record in data:
        if not isinstance(record, dict) or not record.get("id"):
            continue
        model_id = record["id"]
        raw_labels = record.get("labels")
        raw_labels = [str(x) for x in raw_labels] if isinstance(raw_labels, list) else None

        # Tri-state. `labels: []` from the server is NOT the same as no
        # `labels` key at all, and neither is the same as a failed join.
        if raw_labels is None:
            join = LabelJoin.UNKNOWN
            raw_labels = []
        elif raw_labels:
            join = LabelJoin.MATCHED
        else:
            join = LabelJoin.NO_LABELS

        running = loaded_by_name.get(model_id)
        meta = metadata.get(model_id) or {}

        entries.append(LibraryEntry(
            model_id=model_id,
            recipe=record.get("recipe"),
            checkpoint=record.get("checkpoint"),
            downloaded=record.get("downloaded") is True,
            suggested=record.get("suggested") is True,
            update_available=record.get("update_available") is True,
            size=record.get("size") if isinstance(record.get("size"), int) else None,
            max_context_window=(
                record.get("max_context_window")
                if isinstance(record.get("max_context_window"), int) else None
            ),
            labels=sorted(l for l in raw_labels if l in CAPABILITY_LABELS),
            raw_labels=raw_labels,
            label_join=join,
            loaded=running is not None,
            pinned=bool(running and running.get("pinned") is True),
            device=running.get("device") if running else None,
            description=meta.get("description", "") or "",
            favorite=bool(meta.get("favorite")),
            archived=bool(meta.get("archived")),
            tags=sorted(meta.get("tags") or []),
            drift=(
                compare_options(record.get("recipe_options"), running.get("recipe_options"))
                if running else None
            ),
            saved_settings=_settings_from(record.get("recipe_options"), record.get("recipe")),
            running_settings=(
                _settings_from(running.get("recipe_options"), running.get("recipe"))
                if running else None
            ),
        ))

    return entries


def unmatched_loaded_models(
    catalog: dict[str, Any] | None,
    health: dict[str, Any] | None,
) -> list[str]:
    """Loaded models with no catalog entry — i.e. where the F9 join FAILED.

    Surfaced rather than swallowed. A loaded model missing from the library is
    either a join we got wrong or a model Lemonade cannot describe, and both
    mean HELM's capability badges are blind for it. The operator should know
    which rows it cannot vouch for.
    """
    ids = {
        r["id"] for r in ((catalog or {}).get("data") or [])
        if isinstance(r, dict) and r.get("id")
    }
    return sorted(
        m["model_name"] for m in ((health or {}).get("all_models_loaded") or [])
        if isinstance(m, dict) and m.get("model_name") and m["model_name"] not in ids
    )


def search(
    entries: list[LibraryEntry],
    *,
    text: str = "",
    labels: set[str] | None = None,
    tags: set[str] | None = None,
    recipes: set[str] | None = None,
    downloaded: bool | None = None,
    favorite: bool | None = None,
    loaded: bool | None = None,
    include_archived: bool = False,
) -> list[LibraryEntry]:
    """§5.3a's unified search — one query across every axis at once.

    The payoff cases from the spec:
      - "favorited, downloaded, tool-calling models tagged hermes-preferred"
      - "everything tagged uncensored-do-not-deploy" (so it never gets picked
        by accident — the lorablated 8B really does sit in the picker next to
        the standard one)
      - "all NPU models with the embeddings label"

    Filters AND together; each filter's own values OR. Archived models are
    hidden unless asked for — that is the whole point of archiving, and it is
    HELM-local because Lemonade has no equivalent.
    """
    labels = labels or set()
    tags = tags or set()
    recipes = recipes or set()
    needle = text.strip().lower()

    def matches(e: LibraryEntry) -> bool:
        if not include_archived and e.archived:
            return False
        if downloaded is not None and e.downloaded is not downloaded:
            return False
        if favorite is not None and e.favorite is not favorite:
            return False
        if loaded is not None and e.loaded is not loaded:
            return False
        if recipes and e.recipe not in recipes:
            return False
        if labels and not labels.issubset(set(e.labels)):
            return False
        if tags and not tags.issubset(set(e.tags)):
            return False
        if needle:
            haystack = " ".join([
                e.model_id, e.checkpoint or "", e.description, " ".join(e.tags),
            ]).lower()
            if needle not in haystack:
                return False
        return True

    results = [e for e in entries if matches(e)]
    # Favourites first (spec §5.3a: "favourites pin to the top"), then loaded,
    # then downloaded, then alphabetical. The operator's own signal outranks
    # Lemonade's.
    results.sort(key=lambda e: (
        not e.favorite, not e.loaded, not e.downloaded, e.model_id.lower(),
    ))
    return results
