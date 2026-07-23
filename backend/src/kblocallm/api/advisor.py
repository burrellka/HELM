"""Model Advisor API. Spec 14c Part 1.

Two surfaces: a per-model Analyze (Layer 1 facts instantly, Layer 2 + Layer 3
labeled) and an Advisor chat ABOUT a model, anchored to its facts. Every field
returned carries its provenance, so the UI can label local facts, external
metadata, and generated text distinctly — the three-source rule the whole
feature hangs on (§1.0).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..advisor_service import (
    SAVED_KINDS,
    SUB_MODEL_WARN_B,
    AdvisorSettingsStore,
    AdvisorSummaryStore,
    HFMetadataService,
    build_summary_prompt,
    build_synthesis_prompt,
)
from ..domain.advisor import (
    compute_fit,
    decode_quant,
    parse_params,
    quant_size_impact,
)
from ..domain.hf_resolve import RepoConfidence, resolve_repo
from ..domain.pool import extract_pool
from ..domain.system_profile import advisor_hardware_context, extract_system_profile
from ..config import get_settings as get_app_settings
from ..upstream.client import CONTROL, FAST, LemonadeClient, UpstreamError
from ..web_search import web_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advisor", tags=["advisor"])

#: Reasoning models (the 26B is MTP) spend tokens on a hidden reasoning_content
#: channel BEFORE emitting content, so a tight budget starves the answer to
#: empty at finish_reason=length. This is the documented token-budget gotcha.
#: Measured live: the 26B reasons ~1100-1600 tokens on a model-analysis prompt
#: before writing the answer. With the deeper teaching prompt the answer itself
#: is longer too, so the budget is generous — reasoning headroom plus room for
#: a few well-explained paragraphs. The starvation fallback still catches the
#: rare case it runs past even this.
_SYNTHESIS_MAX_TOKENS = 4000
_CHAT_MAX_TOKENS = 3500

#: Summary (§1.10) caps low so a runaway generation cannot stretch the run. The
#: default model (Aevum) is non-reasoning, so unlike the 26B there is no hidden
#: reasoning channel to fund first — a few hundred tokens is a whole 2-4 sentence
#: summary with headroom. Measured: a real starved Summary used ~320 tokens/1.6s.
_SUMMARY_MAX_TOKENS = 500

#: SSE plumbing for the streaming chat. Mirrors operations.py's log stream: a
#: byte must cross the wire every ~15s or an intermediary (NPM's nginx, or a
#: Cloudflare hop) reads the silence of a long generation as a dead origin and
#: kills it. HELM's own nginx already sets `proxy_buffering off` +
#: `proxy_read_timeout 600s`; these headers are the belt to that suspenders.
_KEEPALIVE_SECONDS = 15.0
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _now() -> float:
    return time.monotonic()


async def _sse_with_keepalive(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap a JSON-event source as SSE, emitting a keepalive comment when quiet.

    A byte must cross the wire every `_KEEPALIVE_SECONDS` or an intermediary reads
    the silence as a dead origin and kills it — and the 26B's COLD first token
    lands at ~14s (the fat-context prefill), right on top of that window.

    The subtlety that cost a live debug: you cannot just wrap the source's
    `__anext__` in `asyncio.wait_for`. When the timeout fires it CANCELS that
    coroutine, and the cancellation propagates INTO the generator's in-flight
    httpx read, tearing down the upstream stream — so after the first keepalive
    no token ever arrives. The fix is a producer task draining the source into a
    queue; only the queue GET is cancelled on timeout, never the source itself.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def producer() -> None:
        try:
            async for event in source:
                await queue.put(("data", event))
        except Exception as e:  # noqa: BLE001 — a backstop; source() already
            # wraps UpstreamError into an error event. Anything else still
            # reaches the client as an error, never a silent dropped stream.
            await queue.put(("error", str(e)))
        finally:
            await queue.put((_DONE, None))

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if kind is _DONE:
                break
            if kind == "error":
                yield f"data: {json.dumps({'event': 'error', 'kind': 'stream', 'detail': payload})}\n\n"
                break
            yield f"data: {payload}\n\n"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def _extract_reply(resp: dict | None) -> tuple[str, str | None, bool]:
    """Pull the answer out of a chat response, aware of reasoning models.

    Returns (text, finish_reason, reasoned). `text` is the content channel,
    never the reasoning_content scratchpad — that is the model thinking out
    loud, not its answer, and surfacing it would be its own kind of lie.
    `reasoned` flags that the model produced reasoning but no content, which
    with finish_reason=length means the budget was eaten by reasoning: a
    truncation to report honestly, not a blank to shrug at.
    """
    choices = (resp or {}).get("choices") or []
    if not choices:
        return "", None, False
    message = choices[0].get("message") or {}
    finish = choices[0].get("finish_reason")
    text = (message.get("content") or "").strip()
    reasoned = bool((message.get("reasoning_content") or "").strip())
    return text, finish, reasoned


# ── settings + model picker ──────────────────────────────────────────────────


class SettingsBody(BaseModel):
    analysis_model: str | None = None
    #: The full editable instruction set (UI: "Advisor instructions"). "" (or
    #: "__default__") restores the shipped architect default.
    persona_prompt: str | None = None
    #: The knowledge file. "" (or "__default__") restores the shipped 14d.
    knowledge_primer: str | None = None
    #: Editable web-search provider ("gemini" | "tavily") and model. The KEY is
    #: never here — it stays in env. "" restores the default.
    web_search_provider: str | None = None
    web_search_model: str | None = None
    #: Summary (§1.10): its own model + instruction box, independent of the
    #: Analyze pair. "" (or "__default__") restores the shipped defaults (Aevum;
    #: the Appendix S text).
    summary_model: str | None = None
    summary_prompt: str | None = None


#: Sentinel a client can send to restore a field to its shipped default,
#: rather than to a literal empty string. Also, a plain "" restores (the store
#: treats empty as unset), but the sentinel is explicit about intent.
_RESTORE_DEFAULT = "__default__"


def _resolve_field(value: str | None) -> object:
    """Map a settings-body field to the store's update semantics: unchanged,
    or set (with the restore sentinel and "" both meaning 'clear the override
    so the shipped default shows again')."""
    if value is None:
        return ...
    if value == _RESTORE_DEFAULT:
        return ""
    return value


@router.get("/settings")
async def get_settings(request: Request) -> dict:
    """Current advisor settings plus the analysis-model picker.

    The picker lists every downloaded model the operator owns, each with its
    parameter count and a sub-2B "fabricates" flag (§1.6). The default (the
    resident 26B) is marked. Never a hardcoded id: the default is just the
    default VALUE, and any owned model can be chosen.
    """
    store = AdvisorSettingsStore(request.app.state.db)
    settings = store.get()
    picker = await _analysis_model_options(request.app.state.lemonade)
    # Summary picks from the same owned models, but the "sub-2B fabricates"
    # warning that applies to the ANALYST is wrong here: a small non-reasoning
    # model is the RIGHT tool for a starved-context summary (§1.10). Reframe the
    # list small-first and drop that warning, so Aevum reads as recommended, not
    # discouraged. A very large model is offered but flagged as slow overkill.
    summary_picker = [
        {"id": o["id"], "params_b": o["params_b"], "not_recommended": False,
         "warning": ("Large — slower than needed for a quick summary."
                     if (o["params_b"] or 0) >= 14 else None)}
        for o in sorted(picker, key=lambda o: (o["params_b"] is None, o["params_b"] or 0))
    ]
    app_settings = get_app_settings()
    return {
        "settings": settings,
        "analysis_models": picker,
        "summary_models": summary_picker,
        "operational_prompt_locked": True,
        # Web search status. `enabled` reflects whether the KEY is set (the key
        # itself is never returned — KAIROS's rule). The editable provider/model
        # come from `settings` (DB), not from here.
        "web_search": {
            "enabled": app_settings.web_search_enabled,
            "provider": settings["web_search_provider"],
            "model": settings["web_search_model"],
        },
    }


@router.put("/settings")
def update_settings(body: SettingsBody, request: Request) -> dict:
    """Update the editable settings. The locked safety kernel is a code constant
    and cannot be changed here — no edit can turn off 'do not fabricate'.

    Send "" or "__default__" for the instructions or knowledge file to clear the
    override and restore the shipped architect default.
    """
    store = AdvisorSettingsStore(request.app.state.db)
    return store.update(
        analysis_model=body.analysis_model if body.analysis_model is not None else ...,
        persona_prompt=_resolve_field(body.persona_prompt),
        knowledge_primer=_resolve_field(body.knowledge_primer),
        web_search_provider=_resolve_field(body.web_search_provider),
        web_search_model=_resolve_field(body.web_search_model),
        # summary_model is a plain pick (a model id or unchanged); summary_prompt
        # takes the restore sentinel like the other text boxes.
        summary_model=body.summary_model if body.summary_model is not None else ...,
        summary_prompt=_resolve_field(body.summary_prompt),
    )


@router.post("/knowledge-file")
async def upload_knowledge_file(request: Request, file: UploadFile = File(...)) -> dict:
    """Attach a knowledge file by upload (Kevin's 'attach as a knowledge file').

    Reads the uploaded text and stores it as the operator's knowledge primer,
    overriding the shipped default. Bounded so a stray huge upload cannot bloat
    the synthesis context or the DB.
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="The file is not UTF-8 text.")
    if len(text) > 200_000:
        raise HTTPException(
            status_code=422,
            detail=f"Knowledge file is {len(text):,} chars; keep it under 200,000 so it fits the "
                   f"advisor's context budget.")
    store = AdvisorSettingsStore(request.app.state.db)
    result = store.update(knowledge_primer=text)
    return {"attached": True, "filename": file.filename, "chars": len(text),
            "knowledge_primer_is_default": result["knowledge_primer_is_default"]}


@router.post("/web-search/test")
async def test_web_search(request: Request) -> dict:
    """Run a real search so the operator can verify the key end to end (KAIROS
    §5). Returns ok/error plus a sample, never the key. Uses the operator's
    configured provider/model so the test exercises exactly what analysis will."""
    app_settings = get_app_settings()
    if not app_settings.web_search_enabled:
        return {"ok": False, "error": "No web search key configured (set HELM_WEB_SEARCH_KEY)."}
    advisor = AdvisorSettingsStore(request.app.state.db).get()
    result = await web_search(
        "What is the Qwen3 family of language models?",
        api_key=app_settings.web_search_key,
        provider=advisor["web_search_provider"],
        model=advisor["web_search_model"],
    )
    return {"ok": result.ok, "error": result.error,
            "sample": (result.answer[:300] if result.ok else None),
            "source_count": len(result.sources)}


async def _resolve_analysis_model(client: LemonadeClient, configured: str, is_default: bool) -> str:
    """Make the analysis model portable across boxes.

    An operator's EXPLICIT pick is respected as-is (if it is absent it fails
    honestly — they chose it). But the shipped DEFAULT assumes the 26B, which a
    fresh box may not have; when the default is not present, fall back to the
    largest suitable (>=2B) downloaded model this box actually owns. So HELM
    deployed to a new machine still analyzes without the operator first
    configuring anything.
    """
    if not is_default:
        return configured
    options = await _analysis_model_options(client)
    ids = {o["id"] for o in options}
    if configured in ids:
        return configured
    suitable = [o for o in options if not o["not_recommended"] and o.get("params_b")]
    if suitable:
        return max(suitable, key=lambda o: o["params_b"])["id"]
    return configured  # nothing better; let it fail honestly downstream


async def _resolve_summary_model(client: LemonadeClient, configured: str, is_default: bool) -> str:
    """Portable resolution for the SUMMARY model (§1.10), mirroring the analysis
    one but pulling the OTHER way on the fallback.

    An explicit operator pick is respected as-is. But the shipped default is
    Aevum-0.6B, which a fresh box may not own; when the default is absent, fall
    back to the SMALLEST owned model — Summary wants fast and cheap, and the
    starved context means a tiny model cannot wander. (Analyze falls to the
    largest for judgment; Summary falls to the smallest for speed.)
    """
    if not is_default:
        return configured
    options = await _analysis_model_options(client)
    ids = {o["id"] for o in options}
    if configured in ids:
        return configured
    with_params = [o for o in options if o.get("params_b")]
    if with_params:
        return min(with_params, key=lambda o: o["params_b"])["id"]
    if options:
        return options[0]["id"]
    return configured  # nothing owned; let it fail honestly downstream


async def _analysis_model_options(client: LemonadeClient) -> list[dict]:
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
    except UpstreamError:
        return []
    options = []
    for m in payload.get("data", []):
        if not isinstance(m, dict) or not m.get("downloaded"):
            continue
        params = parse_params(m.get("id"))
        sub = params is not None and params.total_b < SUB_MODEL_WARN_B
        options.append({
            "id": m.get("id"),
            "params_b": params.total_b if params else None,
            "not_recommended": sub,
            "warning": ("Below 2B: fabricates specs in this role. Not recommended as the analyst."
                        if sub else None),
        })
    options.sort(key=lambda o: (o["params_b"] is None, -(o["params_b"] or 0)))
    return options


# ── analyze ──────────────────────────────────────────────────────────────────


class AnalyzeOut(BaseModel):
    model_id: str
    #: Layer 1 — deterministic, no model ran. Always present.
    local_facts: dict[str, Any]
    #: Layer 2 — HuggingFace metadata, or an honest "no external record".
    external: dict[str, Any]
    #: Layer 3 — generated FRESH this call, clearly labeled. Null unless
    #: synthesize was asked for and the analysis model answered.
    synthesis: dict[str, Any] | None = None
    #: A previously-SAVED fast Summary for this model, shown instantly so the
    #: slow generation is not repeated. Null if none saved. Carries its date and
    #: the model that wrote it, because it is a dated snapshot, not a live read.
    saved_summary: dict[str, Any] | None = None
    #: A previously-SAVED Deep analysis, same idea as saved_summary but its own
    #: slot so saving one never clobbers the other. Null if none saved.
    saved_deep: dict[str, Any] | None = None


async def _model_record(client: LemonadeClient, model_id: str) -> dict | None:
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
    except UpstreamError:
        return None
    for m in payload.get("data", []):
        if isinstance(m, dict) and m.get("id") == model_id:
            return m
    return None


async def _pool_and_use(client: LemonadeClient) -> tuple[float | None, float | None]:
    try:
        info = await client.request("GET", "/api/v1/system-info", timeout=FAST)
        stats = await client.request("GET", "/api/v1/system-stats", timeout=FAST)
    except UpstreamError:
        return None, None
    pool = extract_pool(info, stats)
    return pool.capacity_gb, pool.used_gb


async def _hardware_context(client: LemonadeClient) -> dict:
    """The live detected hardware for the advisor to ground in — no baked specs.
    Best-effort: on a read failure the advisor is told the box is undetected
    rather than being handed a stale or invented machine."""
    try:
        info = await client.request("GET", "/api/v1/system-info", timeout=FAST)
        stats = await client.request("GET", "/api/v1/system-stats", timeout=FAST)
    except UpstreamError:
        info, stats = None, None
    return advisor_hardware_context(extract_system_profile(info, stats))


async def _advisor_search(
    model_id: str, facts: dict, resolution, advisor_settings: dict,
    question: str | None = None,
) -> dict | None:
    """Run a grounded web lookup for the advisor (KAIROS §4A direct call).

    Returns None when no key is configured, so the prompt marks search
    "not_run" and the model stays honest. Any search failure comes back as a
    structured result the model can reason about, never a silent blank.

    The KEY is from env (secret); the provider and model come from
    `advisor_settings` (operator-editable, DB-stored), so a swap to a newer
    grounding model takes effect with no redeploy.
    """
    app_settings = get_app_settings()
    if not app_settings.web_search_enabled:
        return None

    repo = getattr(resolution, "repo_id", None)
    subject = repo or model_id
    if question:
        query = f"{subject} language model: {question}"
    else:
        query = (
            f"{subject} language model: what it is, benchmarks, strengths and weaknesses, "
            f"license, recommended use cases and known issues"
        )
    result = await web_search(
        query,
        api_key=app_settings.web_search_key,
        provider=advisor_settings["web_search_provider"],
        model=advisor_settings["web_search_model"],
    )
    if result.ok:
        return {"status": "ok", "query": query, "answer": result.answer, "sources": result.sources}
    return {"status": "failed", "query": query, "error": result.error,
            "note": "A live web lookup was attempted but did not return results. Say so; do not "
                    "fill it from memory."}


async def _owned_inventory(client: LemonadeClient, exclude_id: str) -> list[dict]:
    """The operator's OTHER downloaded models, compact, for §1.5 alternatives.

    This is the candidate set the advisor recommends from: real inventory, so
    it cannot invent a model that is not owned. Kept terse (a handful of fields
    each) so a 25-model fleet does not blow the synthesis context budget.
    """
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
    except UpstreamError:
        return []
    out: list[dict] = []
    for m in payload.get("data", []):
        if not isinstance(m, dict) or not m.get("downloaded") or m.get("id") == exclude_id:
            continue
        params = parse_params(m.get("id"))
        quant = decode_quant(m.get("checkpoint")) or decode_quant(m.get("id"))
        labels = [l for l in (m.get("labels") or []) if l in (
            "tool-calling", "vision", "reasoning", "mtp", "embeddings", "coding")]
        out.append({
            "id": m.get("id"),
            "params_b": params.total_b if params else None,
            "active_b": params.active_b if (params and params.is_moe) else None,
            "quant": quant.label if quant else None,
            "device": "npu" if m.get("recipe") == "flm" else "gpu",
            "size_gb": m.get("size") if isinstance(m.get("size"), (int, float)) else None,
            "capabilities": labels,
        })
    return out


def _local_facts(record: dict, pool_gb: float | None, used_gb: float | None) -> dict[str, Any]:
    """Layer 1 assembly. Pure facts and labeled estimates, no model."""
    model_id = record.get("id") or ""
    checkpoint = record.get("checkpoint")
    size_gb = record.get("size") if isinstance(record.get("size"), (int, float)) else None

    quant = decode_quant(checkpoint) or decode_quant(model_id)
    params = parse_params(model_id) or parse_params(checkpoint)
    fit = compute_fit(size_gb=size_gb, pool_gb=pool_gb, used_gb=used_gb)
    labels = record.get("labels") if isinstance(record.get("labels"), list) else []

    facts: dict[str, Any] = {
        "recipe": record.get("recipe"),
        "device": "npu" if record.get("recipe") == "flm" else "gpu",
        "downloaded": record.get("downloaded") is True,
        "size_gb": size_gb,
        "max_context_window": record.get("max_context_window"),
        "capability_labels": labels,
        "tool_calling": "tool-calling" in labels,
    }
    if params:
        facts["parameters"] = {
            "total_b": params.total_b, "active_b": params.active_b, "moe": params.is_moe,
        }
    if quant:
        facts["quant"] = {
            "label": quant.label, "class": quant.cls.value,
            "bytes_per_weight": quant.bytes_per_weight,
            "explanation": quant.explanation, "tradeoff": quant.tradeoff,
        }
        if size_gb:
            impact = quant_size_impact(size_gb, quant)
            if impact:
                facts["quant"]["size_impact"] = impact
    facts["fit"] = {
        "verdict": fit.verdict.value, "footprint_gb": fit.footprint_gb,
        "measured": fit.measured, "detail": fit.detail,
    }
    return facts


@router.post("/analyze/{model_id:path}", response_model=AnalyzeOut)
async def analyze(model_id: str, request: Request, synthesize: bool = False) -> AnalyzeOut:
    """Analyze one model. Layer 1 always; Layer 2 cached; Layer 3 on request.

    Layer 1 is synchronous and deterministic (no model runs), so the facts card
    renders instantly. Layer 2 is the cached HF lookup. Layer 3 (synthesize=
    true) runs the analysis model over Layers 1+2 only.
    """
    client: LemonadeClient = request.app.state.lemonade
    record = await _model_record(client, model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the catalog.")

    pool_gb, used_gb = await _pool_and_use(client)
    facts = _local_facts(record, pool_gb, used_gb)

    # Layer 2: resolve then fetch. Never guess a repo (§1.8).
    resolution = resolve_repo(record.get("checkpoint"), model_id)
    if resolution.confidence is RepoConfidence.EXPLICIT and resolution.repo_id:
        hf = await HFMetadataService(request.app.state.db).fetch(resolution.repo_id)
        external = {
            "repo_id": resolution.repo_id,
            "resolution": resolution.reason,
            "status": hf.status,
            "cached": hf.cached,
            "metadata": hf.payload if hf.status == "found" else None,
            "note": hf.reason or None,
        }
    else:
        external = {
            "repo_id": None, "resolution": resolution.reason,
            "status": "no_external_record", "metadata": None,
            "note": "Showing local facts only.",
        }

    synthesis = None
    if synthesize:
        synthesis = await _synthesize(request, model_id, facts, resolution, external)

    store = AdvisorSummaryStore(request.app.state.db)
    saved = store.get(model_id, kind="summary")
    saved_deep = store.get(model_id, kind="deep")

    return AnalyzeOut(model_id=model_id, local_facts=facts, external=external,
                      synthesis=synthesis, saved_summary=saved, saved_deep=saved_deep)


@router.post("/summarize/{model_id:path}")
async def summarize(model_id: str, request: Request) -> dict:
    """Summary (14c §1.10): a fast, starved-context read of ONE model.

    Deliberately NOT a lighter call into the synthesis path. It builds its own
    tiny context (the facts card only — no inventory, no web search, no
    knowledge file), runs its OWN model (Aevum by default, operator-pickable),
    and caps tokens low. The result is honest the same way the deep synthesis is
    — labeled generated, grounded to the card, search always off — it just does
    far less work. Target: a couple of seconds, not two minutes.
    """
    client: LemonadeClient = request.app.state.lemonade
    record = await _model_record(client, model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the catalog.")

    pool_gb, used_gb = await _pool_and_use(client)
    facts = _local_facts(record, pool_gb, used_gb)

    settings = AdvisorSettingsStore(request.app.state.db).get()
    summary_model = await _resolve_summary_model(
        client, settings["summary_model"], settings["summary_model_is_default"])
    prompt = build_summary_prompt(settings["summary_prompt"], facts)

    try:
        resp = await client.request(
            "POST", "/api/v1/chat/completions",
            json_body={
                "model": summary_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": (
                        f"Summarize {model_id} in 2-4 plain sentences, using only the facts card."
                    )},
                ],
                "max_tokens": _SUMMARY_MAX_TOKENS, "stream": False,
            },
            timeout=CONTROL,
        )
    except UpstreamError as e:
        return {"generated": True, "available": False, "model": summary_model,
                "note": f"Summary unavailable: {e}. The facts card above stands."}

    text, finish, reasoned = _extract_reply(resp)
    if not text:
        note = "The summary model returned nothing. The facts card above stands."
        if reasoned and finish == "length":
            note = (
                f"{summary_model} spent its token budget reasoning and produced no summary. Pick a "
                f"non-reasoning model for Summary in Advisor settings (Aevum is the default). The "
                f"facts card above is unaffected."
            )
        return {"generated": True, "available": False, "model": summary_model,
                "finish_reason": finish, "note": note}
    return {
        "generated": True, "available": True, "model": summary_model, "text": text,
        "finish_reason": finish,
        "label": f"Generated by {summary_model} from the facts card. Not an independent source.",
    }


class SaveSummaryBody(BaseModel):
    text: str
    analysis_model: str | None = None
    #: Which saved slot: 'summary' (the fast read) or 'deep' (the slow teach).
    #: Defaults to summary for backward compatibility with existing callers.
    kind: str = "summary"


def _valid_kind(kind: str) -> str:
    if kind not in SAVED_KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {SAVED_KINDS}.")
    return kind


@router.post("/summary/{model_id:path}")
def save_summary(model_id: str, body: SaveSummaryBody, request: Request) -> dict:
    """Persist a generated write-up so it shows instantly next time. Summary and
    Deep analysis have separate slots (body.kind), so saving one never clobbers
    the other. Overwrites the prior saved one of the same kind."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Nothing to save.")
    kind = _valid_kind(body.kind)
    store = AdvisorSummaryStore(request.app.state.db)
    store.save(model_id, body.text, body.analysis_model, kind=kind)
    return {"saved": True, **(store.get(model_id, kind=kind) or {})}


@router.delete("/summary/{model_id:path}")
def delete_summary(model_id: str, request: Request, kind: str = "summary") -> dict:
    AdvisorSummaryStore(request.app.state.db).delete(model_id, kind=_valid_kind(kind))
    return {"deleted": True}


async def _synthesize(
    request: Request, model_id: str, facts: dict, resolution, external: dict,
) -> dict[str, Any]:
    """Layer 3. Run the analysis model over Layers 1+2 only, labeled generated.

    On any failure (model not loaded, inference error) it returns a labeled
    'unavailable' rather than a fabricated summary — the whole point is that the
    absence of a synthesis is honest, not papered over."""
    from ..advisor_service import AdvisorSettingsStore, HFResult

    client: LemonadeClient = request.app.state.lemonade
    settings = AdvisorSettingsStore(request.app.state.db).get()
    analysis_model = await _resolve_analysis_model(
        client, settings["analysis_model"], settings["analysis_model_is_default"])
    persona = settings["persona_prompt"]

    hf_result = HFResult(
        status=external.get("status", "no_external_record"),
        payload=external.get("metadata") or {}, cached=bool(external.get("cached")),
        reason=external.get("note") or "",
    )
    owned = await _owned_inventory(client, model_id)
    search = await _advisor_search(model_id, facts, resolution, settings)
    hardware = await _hardware_context(client)
    prompt = build_synthesis_prompt(
        persona, facts, hf_result, owned_models=owned,
        web_search=search, knowledge_primer=settings.get("knowledge_primer", ""),
        hardware=hardware)

    try:
        resp = await client.request(
            "POST", "/api/v1/chat/completions",
            json_body={
                "model": analysis_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": (
                        f"Give a thorough, beginner-friendly analysis of {model_id}. Teach me what "
                        f"it is and what its quantization and size mean in practice; tell me what "
                        f"it is good for and what it is not; explain whether it fits my hardware "
                        f"and why; and if one of my other models would be a better choice for a "
                        f"common use case, say which and why. Explain concepts as you go — I am new "
                        f"to this. Use only the context."
                    )},
                ],
                "max_tokens": _SYNTHESIS_MAX_TOKENS, "stream": False,
            },
            timeout=CONTROL,
        )
    except UpstreamError as e:
        return {"generated": True, "available": False, "model": analysis_model,
                "note": f"Synthesis unavailable: {e}. The local facts and metadata above stand."}

    text, finish, reasoned = _extract_reply(resp)
    if not text:
        note = "The analysis model returned nothing. Local facts above stand."
        if reasoned and finish == "length":
            # The gotcha: reasoning ate the whole budget. Say so, and point at
            # the cheaper non-reasoning analyst.
            note = (
                f"{analysis_model} spent its whole token budget reasoning and produced no summary. "
                f"Try again, or pick a non-reasoning analyst (an 8B) in Advisor settings. The local "
                f"facts above are unaffected."
            )
        return {"generated": True, "available": False, "model": analysis_model,
                "finish_reason": finish, "note": note}
    return {
        "generated": True, "available": True, "model": analysis_model, "text": text,
        "finish_reason": finish,
        "label": f"Generated by {analysis_model} over the facts above. Not an independent source.",
    }


# ── advisor chat (the tab) ───────────────────────────────────────────────────


class ChatBody(BaseModel):
    model_id: str
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)


async def _prepare_chat(request: Request, body: ChatBody) -> tuple[str, list[dict]]:
    """Build the grounded chat request: the resolved model + the messages.

    Shared by the streaming and non-streaming chat endpoints so the two cannot
    drift in what they ground on. Raises 404 if the model is not in the catalog.
    """
    client: LemonadeClient = request.app.state.lemonade
    record = await _model_record(client, body.model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{body.model_id} is not in the catalog.")

    pool_gb, used_gb = await _pool_and_use(client)
    facts = _local_facts(record, pool_gb, used_gb)
    resolution = resolve_repo(record.get("checkpoint"), body.model_id)

    from ..advisor_service import AdvisorSettingsStore, HFResult
    hf = HFResult("no_external_record", {}, cached=False)
    if resolution.confidence is RepoConfidence.EXPLICIT and resolution.repo_id:
        hf = await HFMetadataService(request.app.state.db).fetch(resolution.repo_id)

    settings = AdvisorSettingsStore(request.app.state.db).get()
    owned = await _owned_inventory(client, body.model_id)
    # Chat searches on the operator's actual question, so "what are they saying
    # online about X" runs a real grounded lookup rather than a canned one.
    search = await _advisor_search(body.model_id, facts, resolution, settings, question=body.message)
    hardware = await _hardware_context(client)
    prompt = build_synthesis_prompt(
        settings["persona_prompt"], facts, hf, owned_models=owned,
        web_search=search, knowledge_primer=settings.get("knowledge_primer", ""),
        hardware=hardware)

    analysis_model = await _resolve_analysis_model(
        client, settings["analysis_model"], settings["analysis_model_is_default"])

    messages = [{"role": "system", "content": prompt}]
    # Carry a bounded history so the tab is a conversation, not one-shot.
    for turn in body.history[-8:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": body.message})
    return analysis_model, messages


@router.post("/chat")
async def advisor_chat(body: ChatBody, request: Request) -> dict:
    """Chat ABOUT a model, anchored to its Layer 1 facts + Layer 2 card (§1.4).

    The non-streaming path, kept for callers that want the whole answer at once.
    The UI uses /chat/stream so the operator sees tokens as they land rather than
    staring at a spinner for the ~1-2 min a 26B reasoning answer takes.
    """
    client: LemonadeClient = request.app.state.lemonade
    analysis_model, messages = await _prepare_chat(request, body)
    try:
        resp = await client.request(
            "POST", "/api/v1/chat/completions",
            json_body={"model": analysis_model, "messages": messages,
                       "max_tokens": _CHAT_MAX_TOKENS, "stream": False},
            timeout=CONTROL,
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=f"Analysis model unavailable: {e}")

    text, finish, reasoned = _extract_reply(resp)
    if not text and reasoned and finish == "length":
        text = (
            "(The analysis model used its whole token budget reasoning and did not produce an "
            "answer. Ask again, or switch to a non-reasoning analyst in Advisor settings.)"
        )
    return {"reply": text, "model": analysis_model, "grounded_on": body.model_id,
            "finish_reason": finish}


@router.post("/chat/stream")
async def advisor_chat_stream(body: ChatBody, request: Request) -> StreamingResponse:
    """Streaming chat — the UI path. Same grounded prompt as /chat, but tokens
    are relayed as they arrive so the first words show in a second or two instead
    of after the whole ~1-2 min generation (the fix for the 'so slow' complaint).

    The 26B is a reasoning model: it emits its hidden reasoning channel FIRST,
    then the answer. We relay reasoning as its own `reasoning` event (rendered as
    a live 'thinking' indicator, never as the answer) so the operator sees motion
    immediately, then `text` deltas for the real reply, then a `complete` event
    carrying usage + wall-clock + model for the per-turn line.
    """
    client: LemonadeClient = request.app.state.lemonade
    analysis_model, messages = await _prepare_chat(request, body)

    async def source() -> AsyncIterator[str]:
        started = _now()
        usage: dict | None = None
        saw_text = saw_reasoning = False
        try:
            async for line in client.stream_lines(
                "POST", "/api/v1/chat/completions",
                json_body={
                    "model": analysis_model, "messages": messages,
                    "max_tokens": _CHAT_MAX_TOKENS, "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue  # a partial frame; skip it, never die on one bad line
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    saw_reasoning = True
                    yield json.dumps({"event": "reasoning", "text": reasoning})
                content = delta.get("content")
                if content:
                    saw_text = True
                    yield json.dumps({"event": "text", "text": content})
        except UpstreamError as e:
            # Deliver the failure IN the stream: the connection is open, so an
            # exception here would just drop it and read as "stream ended".
            yield json.dumps({"event": "error", "kind": e.kind, "detail": str(e)})
            return

        done: dict[str, Any] = {
            "event": "complete", "model": analysis_model,
            "grounded_on": body.model_id, "elapsed_s": round(_now() - started, 1),
            "usage": usage or {},
        }
        if not saw_text and saw_reasoning:
            # The reasoning-drain gotcha, live: budget spent thinking, no answer.
            done["note"] = (
                f"{analysis_model} used its whole token budget reasoning and produced no answer. "
                f"Ask again, or switch to a non-reasoning analyst in Advisor settings."
            )
        yield json.dumps(done)

    return StreamingResponse(
        _sse_with_keepalive(source()),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
