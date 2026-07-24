"""Model Advisor Layers 2 and 3, plus settings. Spec 14c Part 1.

Layer 1 (deterministic facts) is pure and lives in `domain/advisor.py`. This
module does the parts that touch the outside world, and it keeps each one
inside the guard the spec sets:

  - **Layer 2 (§1.2)** fetches HuggingFace metadata, SERVER-side only (the
    browser never reaches the web), cached in SQLite by resolved repo id, and
    it degrades to a truthful "no external record" rather than inventing a
    card. A LAN-only control plane does not get arbitrary open-web egress.
  - **Layer 3 (§1.3)** runs the analysis model over Layers 1 and 2 ONLY. Its
    system prompt is two layers: an operational layer LOCKED in code (ground in
    the provided context, never fabricate specs, say unknown) and an editable
    persona layer. The model narrates facts; it is never the source of one.

The operational prompt is the load-bearing safety text and it is a constant
here, not a setting, so no operator edit can turn off "do not fabricate".
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from importlib import resources
from typing import Any

import httpx

from .storage.db import SQLiteManager

logger = logging.getLogger(__name__)


def _load_advisor_data(name: str, fallback: str = "") -> str:
    """Load a shipped default (instructions / knowledge primer) from package data.

    These are the architect's canonical files (14d and Advisor instructions.txt),
    lifted into the package as the SHIPPED DEFAULTS. They are only the starting
    point: the operator edits both from the UI and those edits, stored in the
    DB, win. Nothing here is hardcoded behaviour — it is a seed, not a lock.
    """
    try:
        return (resources.files("kblocallm.advisor_data") / name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as e:
        logger.warning("Advisor default %s not found (%s); using fallback.", name, e)
        return fallback

HF_API = "https://huggingface.co/api/models"

#: Layer 2 cache TTL. HF metadata (downloads, likes, license) moves slowly; a
#: week keeps the Analyze button instant without going stale in any way that
#: matters. 14c §1.2 default.
HF_CACHE_TTL_SECONDS = 7 * 24 * 3600

#: Politeness floor between live HF calls, so a burst of Analyze clicks cannot
#: turn into a burst of outbound requests. Cache hits are exempt.
_MIN_FETCH_INTERVAL_S = 1.0

#: The default analysis model: the resident 26B (14c §1.6). Used when the
#: operator has not chosen one in settings. NOT hardcoded at the call site —
#: it is the default VALUE of an operator-editable setting, which is Kevin's
#: rule: default to the 26B, always allow a pick.
DEFAULT_ANALYSIS_MODEL = "Gemma-4-26B-A4B-it-MTP-GGUF"

#: Below this parameter count a model fabricates specs in this role and must
#: never be the silent default (14c §1.6). The picker keeps them, labeled.
SUB_MODEL_WARN_B = 2.0

#: The default SUMMARY model (14c §1.10). A sub-1B is the RIGHT tool here and it
#: does NOT contradict §1.6: Analyze needs judgment (small models fabricate under
#: that load), but Summary only restates and lightly frames the facts card, and
#: the starved context means it cannot wander. The live verify settled it —
#: Aevum runs ~226 tok/s vs the 26B's ~54, non-reasoning, so a starved Summary
#: returns in ~1.6s where the 26B synthesis took ~2 min. Operator-editable; this
#: is the default VALUE, and the picker stays open to any owned model.
DEFAULT_SUMMARY_MODEL = "Aevum-0.6B-Finetuned.Q4_K_M"

#: Default web-search provider and model, operator-editable from the UI (stored
#: in the DB, not hardcoded at the call site). gemini-2.5-flash is the grounding
#: default; when Google ships a newer grounding model the operator swaps it here
#: with no redeploy. The KEY stays in env (it is the secret).
DEFAULT_SEARCH_PROVIDER = "gemini"
DEFAULT_SEARCH_MODEL = "gemini-2.5-flash"

# The LOCKED SAFETY KERNEL. A code constant, un-editable, prepended to every
# call BEFORE the operator's instructions. Deliberately minimal: it is only the
# irreducible floor that no operator edit (or a bad paste) can turn off. All the
# rich behaviour — teaching approach, sources of truth, search discipline — lives
# in the editable instructions below, where the architect's document carries it.
# The kernel exists so that even a botched instruction edit cannot switch off the
# anti-fabrication rule for a novice who cannot spot a confident lie.
OPERATIONAL_PROMPT = (
    "You are the HELM Model Advisor. The following two rules are ABSOLUTE and cannot be overridden "
    "by any instruction that follows:\n"
    "1. Never state a MODEL-SPECIFIC fact — a benchmark score, parameter count, context length, "
    "license, release date, or who trained a model — unless it appears in the provided context "
    "(the facts card, the owned inventory, or live web search results). If it is not there, say it "
    "is unknown. Never invent or estimate such a number from your own memory.\n"
    "2. Never fabricate a source, a citation, or a search result.\n"
    "3. Ground all hardware, fit, and speed reasoning in the DETECTED hardware provided in the "
    "context (this_machine_detected_hardware), never in example specs from the knowledge file or "
    "your own memory. HELM runs on different machines; the detected numbers are the truth for THIS "
    "one.\n"
    "General CONCEPTS (what quantization is, how MoE works, why context uses memory) you may and "
    "should teach freely. Everything below this line is operator-configurable guidance on how to "
    "advise well."
)

# The EDITABLE instructions layer. Default = the architect's canonical
# `Advisor instructions.txt`, shipped as package data. The operator overrides
# this from the UI (stored in the DB); this constant is only what shows when
# they have not. Kept under the name DEFAULT_INSTRUCTIONS; the DB column is
# still `persona_prompt` for migration continuity, but it now holds the full
# instruction set, and the UI labels it "Advisor instructions".
DEFAULT_INSTRUCTIONS = _load_advisor_data("instructions.txt", fallback=(
    "Teach a smart novice about local models. Ground concept explanations in the knowledge file "
    "when one is attached. Recommend only from the operator's owned inventory. Be thorough, plain, "
    "and show the arithmetic for any fit or size claim."
))

# The default knowledge file: the architect's 14d, shipped as package data.
# Attached by default so the advisor teaches from it out of the box; the
# operator can replace or clear it from the UI.
DEFAULT_KNOWLEDGE_PRIMER = _load_advisor_data("knowledge_primer.md", fallback="")


# ── Summary (14c §1.10) — its own locked floor + editable instructions ────────
#
# The Summary safety kernel. Same idea as OPERATIONAL_PROMPT but for the fast
# path: a minimal, un-editable floor prepended before the operator's summary
# instructions, so even a botched edit cannot switch off "facts card only, no
# lookup, no fabrication" for a novice who cannot spot a confident lie. The rich,
# tunable framing lives in the editable instructions (Appendix S default) below.
SUMMARY_OPERATIONAL_PROMPT = (
    "You are the HELM model Summary. These rules are ABSOLUTE and cannot be overridden by any "
    "instruction that follows:\n"
    "1. Speak ONLY from the facts card provided below. Never state a benchmark score, release "
    "date, training detail, parameter count, or context length that is not on the card. If the "
    "card does not carry something, do not mention it. Never invent or estimate a spec from memory.\n"
    "2. You did NOT look anything up. Never imply you searched or consulted an outside source.\n"
    "3. This is a summary, not an analysis: do not rank or compare against other models, do not walk "
    "fit math, do not recommend alternatives. Keep it to a few plain sentences.\n"
    "Everything below this line is operator-configurable guidance on how to summarize well."
)

# The EDITABLE summary instructions. Default = the architect's Appendix S text,
# shipped as package data. Operator overrides from the UI (DB-stored) win; this
# is only what shows when they have not. Independent of DEFAULT_INSTRUCTIONS,
# which drives Analyze — the two boxes do not share.
DEFAULT_SUMMARY_INSTRUCTIONS = _load_advisor_data("summary_instructions.txt", fallback=(
    "In a few plain sentences, tell a non-expert what this ONE model is and what it is good for, "
    "using only the facts card. Identity first (size, dense-or-MoE, quant, standout capability), "
    "then the kind of task it suits. Gloss any jargon in half a clause. No headers, no lists, no "
    "hype. Shorter is better."
))


@dataclass(frozen=True)
class HFResult:
    status: str  # 'found' | 'not_found' | 'error'
    payload: dict[str, Any]
    cached: bool
    reason: str = ""


class AdvisorSettingsStore:
    """The editable settings (§1.7): analysis model, instructions, knowledge
    file. Each defaults to a shipped value the operator can override from the
    UI; overrides live in the DB and win. The locked safety kernel
    (OPERATIONAL_PROMPT) is the one thing NOT stored here — it cannot be edited."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def get(self) -> dict[str, Any]:
        row = self._db.query_one(
            "SELECT analysis_model, persona_prompt, knowledge_primer, "
            "       web_search_provider, web_search_model, summary_model, summary_prompt "
            "FROM advisor_settings WHERE id = 1")
        model_override = row["analysis_model"] if row else None
        instr_override = row["persona_prompt"] if row else None
        primer_override = row["knowledge_primer"] if row else None
        provider_override = row["web_search_provider"] if row else None
        search_model_override = row["web_search_model"] if row else None
        summary_model_override = row["summary_model"] if row else None
        summary_prompt_override = row["summary_prompt"] if row else None
        return {
            "analysis_model": model_override or DEFAULT_ANALYSIS_MODEL,
            "analysis_model_is_default": not model_override,
            # Named persona_prompt for wire/DB continuity; it is the full
            # editable instruction set, labeled "Advisor instructions" in the UI.
            "persona_prompt": instr_override or DEFAULT_INSTRUCTIONS,
            "persona_is_default": not instr_override,
            "knowledge_primer": primer_override or DEFAULT_KNOWLEDGE_PRIMER,
            "knowledge_primer_is_default": not primer_override,
            # True when a primer is active at all (shipped default OR override),
            # so the UI can say "attached" vs "none".
            "knowledge_primer_active": bool(primer_override or DEFAULT_KNOWLEDGE_PRIMER),
            # Editable search provider + model (the KEY stays in env).
            "web_search_provider": provider_override or DEFAULT_SEARCH_PROVIDER,
            "web_search_model": search_model_override or DEFAULT_SEARCH_MODEL,
            "web_search_model_is_default": not search_model_override,
            # Summary (§1.10): its OWN model + instruction box, independent of
            # the Analyze pair above. Defaults to Aevum + the Appendix S text.
            "summary_model": summary_model_override or DEFAULT_SUMMARY_MODEL,
            "summary_model_is_default": not summary_model_override,
            "summary_prompt": summary_prompt_override or DEFAULT_SUMMARY_INSTRUCTIONS,
            "summary_prompt_is_default": not summary_prompt_override,
        }

    def update(
        self, *,
        analysis_model: str | None = ...,
        persona_prompt: str | None = ...,
        knowledge_primer: str | None = ...,
        web_search_provider: str | None = ...,
        web_search_model: str | None = ...,
        summary_model: str | None = ...,
        summary_prompt: str | None = ...,
    ) -> dict:
        fields = {
            "analysis_model": analysis_model,
            "persona_prompt": persona_prompt,
            "knowledge_primer": knowledge_primer,
            "web_search_provider": web_search_provider,
            "web_search_model": web_search_model,
            "summary_model": summary_model,
            "summary_prompt": summary_prompt,
        }
        with self._db.serialized() as conn:
            conn.execute("INSERT INTO advisor_settings (id) VALUES (1) ON CONFLICT(id) DO NOTHING")
            for column, value in fields.items():
                if value is not ...:
                    conn.execute(
                        f"UPDATE advisor_settings SET {column} = ?, updated_at = datetime('now') "
                        f"WHERE id = 1", (value,))
            conn.commit()
        return self.get()


#: The two kinds of saved write-up a model can carry. 'summary' is the fast read
#: (§1.10); 'deep' is the slow teaching synthesis. Each has its own saved slot so
#: saving one never clobbers the other.
SAVED_KINDS = ("summary", "deep")


class AdvisorSummaryStore:
    """Persisted generated write-ups, keyed by (model_id, kind). The slow ones
    are saved on the operator's say-so and shown instantly on return."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def get(self, model_id: str, kind: str = "summary") -> dict | None:
        row = self._db.query_one(
            "SELECT summary, analysis_model, generated_at FROM advisor_summaries "
            "WHERE model_id = ? AND kind = ?", (model_id, kind))
        if row is None:
            return None
        return {"text": row["summary"], "analysis_model": row["analysis_model"],
                "generated_at": row["generated_at"], "kind": kind, "saved": True}

    def save(self, model_id: str, summary: str, analysis_model: str | None,
             kind: str = "summary") -> None:
        self._db.execute(
            "INSERT INTO advisor_summaries (model_id, kind, summary, analysis_model, generated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(model_id, kind) DO UPDATE SET summary = excluded.summary, "
            "  analysis_model = excluded.analysis_model, generated_at = excluded.generated_at",
            (model_id, kind, summary, analysis_model))

    def delete(self, model_id: str, kind: str = "summary") -> None:
        self._db.execute(
            "DELETE FROM advisor_summaries WHERE model_id = ? AND kind = ?", (model_id, kind))


class ModelSourceStore:
    """Remembers the HuggingFace checkpoint a model was pulled from (migration
    11). Lemonade reports a local file path post-download; this keeps the origin
    so the advisor's external-metadata lookup can resolve the real repo."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db

    def get(self, model_id: str) -> str | None:
        row = self._db.query_one(
            "SELECT checkpoint FROM model_source WHERE model_id = ?", (model_id,))
        return row["checkpoint"] if row else None

    def set(self, model_id: str, checkpoint: str, recipe: str | None = None) -> None:
        self._db.execute(
            "INSERT INTO model_source (model_id, checkpoint, recipe, recorded_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(model_id) DO UPDATE SET checkpoint = excluded.checkpoint, "
            "  recipe = excluded.recipe, recorded_at = excluded.recorded_at",
            (model_id, checkpoint, recipe))


class HFMetadataService:
    """Layer 2. Server-side, cached, rate-limited, honest about absence."""

    def __init__(self, db: SQLiteManager) -> None:
        self._db = db
        self._last_fetch = 0.0

    async def fetch(self, repo_id: str, *, force: bool = False) -> HFResult:
        if not force:
            cached = self._read_cache(repo_id)
            if cached is not None:
                return cached

        # Rate-limit live calls (cache hits already returned above).
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < _MIN_FETCH_INTERVAL_S:
            time.sleep(_MIN_FETCH_INTERVAL_S - elapsed)
        self._last_fetch = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(
                    f"{HF_API}/{repo_id}",
                    headers={"User-Agent": "HELM-control-plane"},
                )
        except httpx.HTTPError as e:
            # A network failure is not "no record"; it is "could not check".
            return HFResult("error", {}, cached=False, reason=f"HuggingFace unreachable: {e}")

        if resp.status_code == 404:
            self._write_cache(repo_id, "not_found", {})
            return HFResult("not_found", {}, cached=False,
                            reason="HuggingFace has no public record for this repo.")
        if resp.status_code != 200:
            return HFResult("error", {}, cached=False,
                            reason=f"HuggingFace returned {resp.status_code}.")

        payload = _distill(resp.json())
        self._write_cache(repo_id, "found", payload)
        return HFResult("found", payload, cached=False)

    def _read_cache(self, repo_id: str) -> HFResult | None:
        row = self._db.query_one(
            "SELECT status, payload, fetched_at, "
            "  (julianday('now') - julianday(fetched_at)) * 86400 AS age_s "
            "FROM hf_cache WHERE repo_id = ?", (repo_id,))
        if row is None or row["age_s"] > HF_CACHE_TTL_SECONDS:
            return None
        return HFResult(row["status"], json.loads(row["payload"]), cached=True)

    def _write_cache(self, repo_id: str, status: str, payload: dict) -> None:
        self._db.execute(
            "INSERT INTO hf_cache (repo_id, status, payload, fetched_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(repo_id) DO UPDATE SET status = excluded.status, "
            "  payload = excluded.payload, fetched_at = excluded.fetched_at",
            (repo_id, status, json.dumps(payload)))


def _distill(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the structured signal, drop the noise. 14c §1.2: structured source,
    not scraped vibes."""
    card = raw.get("cardData") or {}
    return {
        "id": raw.get("id"),
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "license": card.get("license") or raw.get("license"),
        "pipeline_tag": raw.get("pipeline_tag"),
        "tags": (raw.get("tags") or [])[:20],
        "last_modified": raw.get("lastModified"),
        "gated": raw.get("gated"),
        # A short slice of the card description if present. Bounded so a huge
        # card cannot blow the synthesis context.
        "description": (card.get("summary") or card.get("description") or "")[:1200],
    }


def build_synthesis_prompt(
    persona: str,
    facts: dict[str, Any],
    hf: HFResult,
    *,
    owned_models: list[dict[str, Any]] | None = None,
    web_search: dict[str, Any] | None = None,
    knowledge_primer: str = "",
    hardware: dict[str, Any] | None = None,
) -> str:
    """Assemble the Layer 3 context. Every source is grounded; the model adds
    nothing from memory.

    `owned_models` is the operator's real inventory (§1.5) so the advisor can
    recommend alternatives WITHOUT inventing a model — the candidate set is
    their actual downloaded models. `web_search` is the grounded-search source
    (Gemini grounding via the KAIROS pattern) once wired; until then it is
    None and the model is told search was not run, never that "nothing was
    found".
    """
    context: dict[str, Any] = {
        # DETECTED hardware first: the advisor grounds fit/speed reasoning in
        # the live box, so HELM is portable across machines (no baked specs).
        "this_machine_detected_hardware": hardware or {
            "status": "unknown",
            "note": "Hardware was not detected this call; do not assume specs.",
        },
        "target_model_local_facts": facts,
        "target_model_external_metadata": (
            hf.payload if hf.status == "found"
            else {"status": hf.status, "note": hf.reason or "No external metadata available."}
        ),
    }
    if owned_models is not None:
        context["operators_other_owned_models"] = (
            owned_models
            or "The operator owns no other downloaded models to compare against."
        )
    if web_search is not None:
        # Real grounded search results. Distinct source, cited.
        context["web_search_results"] = web_search
    else:
        context["web_search_results"] = {
            "status": "not_run",
            "note": "Web search was not run for this analysis. Do not substitute your own "
                    "recollection for it; say a live lookup was not performed if the operator "
                    "asks what is being said online.",
        }

    # The vetted primer goes in as its own labeled section rather than buried in
    # the JSON, so the model treats it as authoritative teaching material.
    primer_block = ""
    if knowledge_primer.strip():
        primer_block = (
            "\n\nREFERENCE PRIMER (authoritative, operator-vetted — ground your explanations of "
            "how local models work in this, not your own recollection):\n"
            f"{knowledge_primer.strip()}\n"
        )

    return (
        f"{OPERATIONAL_PROMPT}\n\n"
        f"ADVISOR INSTRUCTIONS:\n{persona}"
        f"{primer_block}\n\n"
        f"CONTEXT (the ONLY information you may use; everything model-specific must come from "
        f"here):\n{json.dumps(context, indent=2, default=str)}"
    )


def build_summary_prompt(instructions: str, facts: dict[str, Any]) -> str:
    """Assemble the STARVED Summary context (14c §1.10). Facts card ONLY.

    Deliberately the opposite of build_synthesis_prompt: no inventory, no web
    search, no knowledge primer, no external metadata, not even hardware. Just
    the one model's local facts card behind the locked summary floor and the
    editable instructions. The starvation is what keeps a sub-1B model grounded
    (it cannot reach for anything that is not on the card) and what keeps the
    output short — the speed itself comes from the small model + a low token cap
    (the live verify showed the fat context's prefill was cheap; generation was
    the wall).
    """
    return (
        f"{SUMMARY_OPERATIONAL_PROMPT}\n\n"
        f"SUMMARY INSTRUCTIONS:\n{instructions}\n\n"
        f"FACTS CARD (the ONLY information you may use):\n"
        f"{json.dumps(facts, indent=2, default=str)}"
    )
