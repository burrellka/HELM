"""Phase 4: operations. Log stream, warm-on-startup, benchmarks. Spec §5.6, §5.7, Rule 5.

Three features, one theme: each is a place where the obvious implementation is
dangerous.

**The log stream will 524 if you just return an iterator** (F5). Cloudflare
kills a response after ~100s with no bytes — an *idle* timer, and a quiet log
tail at 3am is exactly that. NPM buffers by default, so the stream never
reaches the edge at all. Both halves are handled here; the NPM half is in
`infra/npm-mtls-ingress.md` and cannot be fixed from this file.

**Warm-on-startup is a write action triggered by an inferred state**, which the
review called the most dangerous item in §6. HELM does NOT infer. See `warm()`.

**Benchmarks run real inference on a shared box.** Operator-triggered only,
never scheduled, never on load.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..domain.load_plan import NPU_RECIPES, LoadIntent, plan_load
from ..domain.logs import classify
from ..storage.metadata import WarmSetStore
from ..upstream.client import COMPLETION, CONTROL, FAST, LemonadeClient, UpstreamError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/operations", tags=["operations"])

#: Cloudflare 524s after ~100s of silence. 15s leaves a wide margin and costs
#: one comment line per interval. This single line is what keeps the log tail
#: alive at 3am (F5).
KEEPALIVE_SECONDS = 15.0

#: Streaming responses must say this or nginx (NPM's, and ours) buffers them
#: into uselessness. `Cache-Control: no-cache` stops any intermediary caching
#: a stream that by definition never repeats.
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


async def _sse_with_keepalive(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap an upstream line iterator as SSE, emitting keepalives when quiet.

    The race is the point: wait for the next upstream line, but never longer
    than KEEPALIVE_SECONDS without putting a byte on the wire. Without this,
    a silent stream is indistinguishable from a dead one to Cloudflare, and it
    kills it.

    Lines beginning with ':' are SSE comments. The frontend's `streamLines`
    already skips them, so the operator never sees the keepalive.
    """
    iterator = source.__aiter__()
    while True:
        try:
            line = await asyncio.wait_for(iterator.__anext__(), timeout=KEEPALIVE_SECONDS)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        except StopAsyncIteration:
            break
        except UpstreamError as e:
            # Deliver the failure IN the stream. The connection is already
            # open, so an exception here would just drop it and the UI would
            # show "stream ended" for what is really "lemond refused".
            yield f"data: {_json({'error': e.kind, 'detail': str(e)})}\n\n"
            break
        yield f"data: {line}\n\n"


def _json(value: Any) -> str:
    import json
    return json.dumps(value)


@router.get("/logs/stream")
async def logs_stream(request: Request) -> StreamingResponse:
    """Live log tail (§5.6).

    Returns raw lines; the client classifies them for highlighting via the
    same `domain.logs.classify` rules exposed at /logs/classify. Kept raw on
    the wire so the pane can show exactly what lemond said — a control plane
    that paraphrases its own logs is not one you can debug with.
    """
    client: LemonadeClient = request.app.state.lemonade

    async def source() -> AsyncIterator[str]:
        async for line in client.stream_lines("GET", "/api/v1/logs/stream"):
            yield line

    return StreamingResponse(
        _sse_with_keepalive(source()),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


class ClassifyBody(BaseModel):
    lines: list[str]


@router.post("/logs/classify")
def classify_lines(body: ClassifyBody) -> dict:
    """Classify log lines (§5.6 highlighting).

    Server-side so the rules are the tested ones — the mass-evict detection in
    particular is order-sensitive, and a reimplementation in TypeScript would
    quietly read the nuclear option as routine housekeeping.
    """
    return {
        "lines": [
            {"text": c.text, "event": c.event.value, "note": c.note}
            for c in (classify(line) for line in body.lines)
        ]
    }


# ── warm on startup (Rule 5 / F7) ───────────────────────────────────────────


class WarmSetBody(BaseModel):
    model_id: str
    note: str = ""


@router.get("/warm-set")
def get_warm_set(request: Request) -> dict:
    """The explicit warm set. Empty by default, and that is deliberate."""
    return {
        "models": WarmSetStore(request.app.state.db).all(),
        "note": (
            "Warming is operator-triggered only. Lemonade exposes no restart signal "
            "(confirmed: neither /health nor /system-info carries uptime or a resetting "
            "monotonic), and an empty model list is ALSO what the router shows mid-incident "
            "while it clears hardware state — so HELM will never infer a restart and "
            "mass-load into it."
        ),
    }


@router.post("/warm-set")
def add_to_warm_set(body: WarmSetBody, request: Request) -> dict:
    WarmSetStore(request.app.state.db).add(body.model_id, body.note)
    return {"models": WarmSetStore(request.app.state.db).all()}


@router.delete("/warm-set/{model_id:path}")
def remove_from_warm_set(model_id: str, request: Request) -> dict:
    WarmSetStore(request.app.state.db).remove(model_id)
    return {"models": WarmSetStore(request.app.state.db).all()}


@router.post("/warm")
async def warm(request: Request, confirm: bool = False) -> dict:
    """Load the warm set. **Operator-triggered ONLY.**

    Rule 5 identifies a real gap: pinned does NOT mean auto-load, so after
    `systemctl restart lemond` everything is cold even though it is still
    marked pinned. Fixing it is an obvious value-add.

    Doing it automatically is the most dangerous idea in the spec, and HELM
    does not:

    - **There is no restart signal.** 14a RESULTS §3 confirmed it: no uptime,
      no start time, no resetting monotonic. The only available inference is
      `all_models_loaded == []`, which is ALSO the nuclear option's signature
      (§3) — a backend failure making the router deliberately evict everything
      to clear hardware state. HELM would watch the router frantically
      clearing hardware, conclude "looks like a restart!", and start
      mass-loading into it. That is not a race, it is HELM fighting the router
      during an incident.
    - **NPU models are never warmed.** Exclusivity means warming one evicts the
      incumbent (Rule 7), unattended, with nobody present to see the warning.
      They are skipped and reported, not silently dropped.
    - **Single-flight.** Two tabs, or a tab and a scheduled job, must not both
      warm.

    So: no scheduler, no startup hook, no inference. A human presses the
    button, having read the room.
    """
    client: LemonadeClient = request.app.state.lemonade
    if not client.has_admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HELM has no Lemonade admin key, so warming is unavailable. "
                "Set HELM_LEMONADE_ADMIN_API_KEY in the stack's .env and restart HELM."
            ),
        )

    lock: asyncio.Lock = _warm_lock(request)
    if lock.locked():
        raise HTTPException(status_code=409, detail={
            "code": "warm_in_progress",
            "message": "A warm run is already in progress. Warming twice concurrently would "
                       "race the router's own eviction logic.",
        })

    wanted = WarmSetStore(request.app.state.db).all()
    if not wanted:
        return {"warmed": [], "skipped": [], "message": (
            "The warm set is empty. Add models to it explicitly — HELM deliberately does not "
            "warm 'everything that was pinned', because pinned means 'do not evict', which is "
            "a different intent from 'bring this back after a restart'."
        )}

    async with lock:
        try:
            health = await client.request("GET", "/api/v1/health", timeout=FAST)
        except UpstreamError as e:
            raise HTTPException(status_code=502, detail=f"Cannot read fleet state: {e}")

        loaded = {
            m.get("model_name") for m in (health.get("all_models_loaded") or [])
            if isinstance(m, dict)
        }
        catalog = await _catalog_recipes(client)

        plan: list[dict] = []
        skipped: list[dict] = []
        for model_id in wanted:
            if model_id in loaded:
                skipped.append({"model_id": model_id, "reason": "already loaded"})
                continue
            recipe = catalog.get(model_id)
            if recipe in NPU_RECIPES:
                # Never unattended. This is a refusal, not an oversight.
                skipped.append({"model_id": model_id, "reason": (
                    "NPU model — warming it would evict whichever NPU model is resident, with "
                    "nobody watching. Load it deliberately from the dashboard instead."
                )})
                continue
            plan.append({"model_id": model_id, "recipe": recipe})

        if not confirm:
            raise HTTPException(status_code=409, detail={
                "code": "needs_confirmation",
                "message": (
                    f"This will load {len(plan)} model(s) onto the fleet. If lemond is "
                    f"mid-incident (the router evicts everything after a backend failure), "
                    f"loading into it makes things worse. Confirm you know the box is idle."
                ),
                "plan": plan, "skipped": skipped,
            })

        warmed: list[dict] = []
        for item in plan:
            body = plan_load(
                LoadIntent(model_name=item["model_id"], recipe=item["recipe"]),
            ).body
            try:
                await client.request(
                    "POST", "/api/v1/load", admin=True, json_body=body, timeout=CONTROL,
                )
                warmed.append({"model_id": item["model_id"], "ok": True})
            except UpstreamError as e:
                # Keep going: one model failing to warm should not abandon the
                # rest, and the operator needs to know WHICH failed.
                logger.warning("Warm of %s failed: %s", item["model_id"], e)
                warmed.append({"model_id": item["model_id"], "ok": False,
                               "error_kind": e.kind, "error": str(e)})

        return {"warmed": warmed, "skipped": skipped}


def _warm_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "warm_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.warm_lock = lock
    return lock


async def _catalog_recipes(client: LemonadeClient) -> dict[str, str]:
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST,
        )
    except UpstreamError:
        return {}
    return {
        r["id"]: r.get("recipe")
        for r in (payload or {}).get("data", [])
        if isinstance(r, dict) and r.get("id")
    }


# ── benchmarks (§5.7) ───────────────────────────────────────────────────────

#: Deliberately small and fixed. §5.7's scenarios are JSON in the CLI harness;
#: HELM's job here is comparable numbers stored over time, not a scenario
#: authoring tool.
SCENARIOS: dict[str, str] = {
    "chat": "Explain what a memory pool is in two sentences.",
    "coding": "Write a Python function that reverses a linked list. Code only.",
    "long-context": "Summarise the key idea in one sentence: " + ("context padding. " * 200),
}


class BenchBody(BaseModel):
    model_name: str
    scenario: str = "chat"
    max_tokens: int = 128


class BenchOut(BaseModel):
    model_name: str
    scenario: str
    ok: bool
    tokens_per_second: float | None = None
    time_to_first_token: float | None = None
    output_tokens: int | None = None
    config_note: str = ""
    error: str | None = None


@router.get("/benchmarks")
def benchmark_history(request: Request, model_id: str | None = None) -> dict:
    """Stored results (§5.7): make model choices evidence-based, not folklore."""
    sql = (
        "SELECT model_id, scenario, started_at, tokens_per_second, time_to_first_token, "
        "       output_tokens, ok, error, config_note FROM benchmark_runs"
    )
    params: tuple = ()
    if model_id:
        sql += " WHERE model_id = ?"
        params = (model_id,)
    sql += " ORDER BY started_at DESC LIMIT 200"
    return {"runs": [dict(r) for r in request.app.state.db.execute(sql, params)]}


@router.post("/benchmark", response_model=BenchOut)
async def benchmark(body: BenchBody, request: Request) -> BenchOut:
    """Run one scenario against a loaded model and store the result.

    **This runs real inference on the shared box**, so it is operator-triggered
    only — never on load, never scheduled, never a side effect of opening a
    page. Six agents and three humans are on this hardware.

    Timing is measured HERE rather than read from /api/v1/stats, because that
    endpoint reports the last request globally (any client's) and can report a
    fabricated 1,000,000 tok/s for a short response. A benchmark that reads a
    global mutable counter is measuring the wrong thing.
    """
    client: LemonadeClient = request.app.state.lemonade
    if body.scenario not in SCENARIOS:
        raise HTTPException(status_code=422, detail=(
            f"Unknown scenario {body.scenario!r}. Available: {', '.join(SCENARIOS)}."
        ))

    config_note = await _config_note(client, body.model_name)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    try:
        result = await client.request(
            "POST", "/api/v1/chat/completions",
            json_body={
                "model": body.model_name,
                "messages": [{"role": "user", "content": SCENARIOS[body.scenario]}],
                "max_tokens": body.max_tokens,
                "stream": False,
                # A throughput probe must measure the model ANSWERING, not
                # thinking. Post-2026.42.1 thinking defaults ON, so a short
                # max_tokens budget would be spent in the reasoning channel and
                # return empty — a meaningless benchmark. Pin it off (nested; the
                # top-level flag is ignored) for a clean, comparable measurement.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=COMPLETION,
        )
    except UpstreamError as e:
        _store_run(request, body, started, None, config_note, error=str(e))
        return BenchOut(
            model_name=body.model_name, scenario=body.scenario, ok=False,
            config_note=config_note, error=str(e),
        )

    elapsed = time.monotonic() - t0
    usage = (result or {}).get("usage") or {}
    output_tokens = usage.get("completion_tokens")
    rate = (
        output_tokens / elapsed
        if isinstance(output_tokens, int) and output_tokens > 0 and elapsed > 0
        else None
    )
    out = BenchOut(
        model_name=body.model_name, scenario=body.scenario, ok=True,
        tokens_per_second=round(rate, 2) if rate else None,
        time_to_first_token=None,  # not separable without streaming; do not invent it
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        config_note=config_note,
    )
    _store_run(request, body, started, out, config_note)
    return out


async def _config_note(client: LemonadeClient, model_name: str) -> str:
    """What the model was running WHEN measured.

    A tok/s figure without its slot and context configuration is not
    comparable to the next one — and comparing them anyway is exactly how
    folklore forms. Stored beside every result.
    """
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        return "configuration unknown (could not read /health at measurement time)"
    for model in health.get("all_models_loaded") or []:
        if isinstance(model, dict) and model.get("model_name") == model_name:
            options = model.get("recipe_options") or {}
            return (
                f"recipe={model.get('recipe')} backend={options.get('llamacpp_backend')} "
                f"ctx_size={options.get('ctx_size')} args={options.get('llamacpp_args')!r}"
            )
    return "model not loaded at measurement time"


def _store_run(
    request: Request, body: BenchBody, started: datetime,
    out: BenchOut | None, config_note: str, error: str = "",
) -> None:
    request.app.state.db.execute(
        "INSERT INTO benchmark_runs (model_id, scenario, started_at, tokens_per_second, "
        "  time_to_first_token, output_tokens, ok, error, config_note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            body.model_name, body.scenario, started.isoformat(),
            out.tokens_per_second if out else None,
            out.time_to_first_token if out else None,
            out.output_tokens if out else None,
            1 if (out and out.ok) else 0,
            error or (out.error if out else "") or "",
            config_note,
        ),
    )
