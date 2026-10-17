"""Phase 2: control. load / unload / pin. Spec §5.3, §5.4.

**Every write in HELM enters through this module.** It is mounted with
`_AUTH_REQUIRED` like everything else (F1a), and it refuses before the network
call when there is no admin key — a clear local error beats a remote 401 the
operator has to interpret (URSA 1.1.8).

The pattern each endpoint follows:

    plan -> if blocked, 422 with the reasons -> else send -> report honestly

The planning lives in `domain.load_plan`, pure and tested. This file does
HTTP. That split is deliberate: the rules that stop an operator evicting the
26B by accident should be provable without a running server.

**`?confirm=true`.** Anything with a WARNING (an NPU eviction, unloading a
pinned model) requires an explicit confirm. The first call returns the plan and
performs nothing; the UI shows the warnings; the second call carries the
confirmation. A control plane should not turn a click into an eviction with no
step in between.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..domain.llamacpp_args import parallel_slots
from ..domain.load_plan import LoadIntent, LoadPlan, Level, plan_load, plan_unload
from ..upstream.client import CONTROL, FAST, LemonadeClient, UpstreamError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/control", tags=["control"])


class IssueOut(BaseModel):
    level: str
    code: str
    message: str


class PlanOut(BaseModel):
    """What HELM would do, and what it would cost."""

    blocked: bool
    #: True when there are warnings the operator has not confirmed yet. The UI
    #: shows them and asks; it does not proceed.
    needs_confirmation: bool
    issues: list[IssueOut] = Field(default_factory=list)
    evicts: list[str] = Field(default_factory=list)
    computed_ctx_size: int | None = None
    #: Exactly what would be POSTed upstream. Shown in the UI's "what will
    #: this do" detail — a control plane the operator cannot audit is one they
    #: are right not to trust.
    body: dict[str, Any] = Field(default_factory=dict)


class ActionOut(BaseModel):
    performed: bool
    plan: PlanOut
    result: Any | None = None
    #: Populated when the upstream refused. Keeps the KIND (house rule #5).
    error_kind: str | None = None
    error_detail: str | None = None
    #: Read-back verification of a load's OPTIONS (slots / ctx). `performed`
    #: only means "Lemonade accepted the reload" — it can accept the call and
    #: come back running DIFFERENT options (e.g. dropping --parallel). None when
    #: there was nothing to verify (no recipe_options sent) or the read-back
    #: itself failed; True/False otherwise. This is the guard that turns a silent
    #: no-op into a visible failure.
    verified: bool | None = None
    verify_note: str | None = None
    #: Models that were resident BEFORE this action and are gone AFTER it. On a
    #: FAILED load this is pure loss — Lemonade evicts to make room before the
    #: load, so a load that then fails takes other models (pinned included) down
    #: with it (confirmed on the box, 3x). Populated only when a load fails and
    #: eviction is detected; empty otherwise.
    evicted: list[str] = Field(default_factory=list)
    #: The one-click restore set for `evicted`: each model with the pinned state
    #: it had, to reload by name (Lemonade re-applies its saved recipe). The UI
    #: offers this; it does not auto-reload, because each reload is 10-20s.
    restorable: list[dict[str, Any]] = Field(default_factory=list)


class LoadBody(BaseModel):
    model_name: str
    recipe: str | None = None
    #: llamacpp only. HELM sends it as `--parallel N`.
    slots: int | None = None
    #: PER SLOT. HELM computes the total; the operator never multiplies.
    ctx_per_slot: int | None = None
    llamacpp_backend: str | None = None
    pinned: bool = False
    #: Sampling params from labeled fields. None = leave unset (model default).
    #: HELM composes these into llamacpp_args; the operator never types the flags.
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    extra_args: str = ""
    save_options: bool = False


class UnloadBody(BaseModel):
    model_name: str


def _out(plan: LoadPlan, confirmed: bool) -> PlanOut:
    warnings = [i for i in plan.issues if i.level is Level.WARNING]
    return PlanOut(
        blocked=plan.blocked,
        needs_confirmation=bool(warnings) and not confirmed,
        issues=[IssueOut(level=i.level.value, code=i.code, message=i.message) for i in plan.issues],
        evicts=plan.evicts,
        computed_ctx_size=plan.computed_ctx_size,
        body=plan.body,
    )


def _client(request: Request) -> LemonadeClient:
    client: LemonadeClient = request.app.state.lemonade
    if not client.has_admin_key:
        # Refuse locally and say why. The UI should already be hiding these
        # controls (management_enabled=false); if we got here, the operator
        # deserves the real reason rather than a bare 401 from Lemonade.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HELM has no Lemonade admin key, so management actions are unavailable. "
                "Set HELM_LEMONADE_ADMIN_API_KEY in the stack's .env and restart HELM."
            ),
        )
    return client


async def _fleet(client: LemonadeClient) -> dict:
    """Current health, for the planner's cross-checks.

    Fetched fresh per action rather than reused from the dashboard poll: the
    NPU-exclusivity check must run against the fleet as it is NOW, not as it
    was up to five seconds ago. In that window someone else's load could have
    changed which model the router is about to evict.
    """
    try:
        return await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not read the current fleet state, so HELM cannot tell you what this "
                f"action would evict. Refusing rather than acting blind. ({e})"
            ),
        )


async def _catalog(client: LemonadeClient, model_name: str) -> dict:
    """The model's catalog record, for `max_context_window`.

    Best-effort: a miss costs the ceiling cross-check, not the action. F9
    confirmed health.model_name == models.id, so this join is safe.
    """
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST,
        )
    except UpstreamError:
        return {}
    for entry in (payload or {}).get("data", []):
        if isinstance(entry, dict) and entry.get("id") == model_name:
            return entry
    return {}


@router.post("/plan/load", response_model=PlanOut)
async def preview_load(body: LoadBody, request: Request) -> PlanOut:
    """Dry run. Performs nothing.

    This is what the load dialog calls on every keystroke-ish change, so the
    operator sees "this will unload qwen3.5-2b-FLM" BEFORE the button, not
    after the click.
    """
    client = _client(request)
    health = await _fleet(client)
    plan = plan_load(
        LoadIntent(**body.model_dump()),
        health=health,
        catalog=await _catalog(client, body.model_name),
    )
    return _out(plan, confirmed=False)


@router.post("/load", response_model=ActionOut)
async def load(body: LoadBody, request: Request, confirm: bool = False) -> ActionOut:
    client = _client(request)
    health = await _fleet(client)
    plan = plan_load(
        LoadIntent(**body.model_dump()),
        health=health,
        catalog=await _catalog(client, body.model_name),
    )
    out = _out(plan, confirmed=confirm)

    if plan.blocked:
        # 422: the request is well-formed but the intent is invalid. The body
        # carries WHY, per issue, so the UI can point at the field.
        raise HTTPException(status_code=422, detail=out.model_dump())
    if out.needs_confirmation:
        # 409: not an error, a gate. "Yes I know it will unload the NPU model."
        raise HTTPException(status_code=409, detail=out.model_dump())

    if plan.evicts:
        logger.warning(
            "Operator load of %s will evict NPU incumbent(s): %s",
            body.model_name, ", ".join(plan.evicts),
        )

    # Snapshot the resident fleet BEFORE the load. Lemonade evicts to make room
    # before the load completes, so if the load fails the evicted models (pinned
    # included) are simply gone. We capture who was loaded, with their pinned
    # state, so a failed load can offer a one-click restore instead of leaving
    # the operator to notice the fleet is missing.
    before = _resident(health, exclude=body.model_name)

    action = await _perform(client, "/api/v1/load", plan.body, out)

    if not action.performed:
        # The load failed. Anything that was resident and is now gone was
        # evicted to make room for a load that never landed — pure loss.
        evicted = await _detect_evicted(client, before)
        if evicted:
            logger.warning(
                "Failed load of %s evicted resident model(s): %s",
                body.model_name, ", ".join(e["model_name"] for e in evicted),
            )
            action.evicted = [e["model_name"] for e in evicted]
            action.restorable = evicted
        return action

    # 'performed' means Lemonade accepted the reload — NOT that it applied our
    # options. It can accept the call and reload with different args. Read back
    # and say so, rather than reporting a silent no-op as success. Only when we
    # actually sent options (top-level fields) to check.
    if any(k in plan.body for k in ("ctx_size", "llamacpp_args")):
        action.verified, action.verify_note = await _verify_load(
            client, body.model_name, plan.body)
    return action


def _resident(health: dict, *, exclude: str | None = None) -> list[dict[str, Any]]:
    """The currently-loaded models as [{model_name, pinned}], minus `exclude`."""
    out: list[dict[str, Any]] = []
    for m in (health.get("all_models_loaded") or []):
        if not isinstance(m, dict):
            continue
        name = m.get("model_name")
        if name and name != exclude and m.get("loaded") is not False:
            out.append({"model_name": name, "pinned": bool(m.get("pinned"))})
    return out


async def _detect_evicted(
    client: LemonadeClient, before: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Which snapshotted models are no longer resident. A read failure returns
    nothing — we never claim an eviction we cannot see."""
    if not before:
        return []
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        return []
    still = {
        m.get("model_name") for m in (health.get("all_models_loaded") or [])
        if isinstance(m, dict) and m.get("loaded") is not False
    }
    return [m for m in before if m["model_name"] not in still]


class RestoreBody(BaseModel):
    #: Each {model_name, pinned} from a failed load's `restorable`.
    models: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/restore")
async def restore(body: RestoreBody, request: Request) -> dict:
    """Reload models evicted by a failed load. Each by NAME, so Lemonade
    re-applies its saved recipe — no options re-sent from HELM, nothing to get
    wrong. Pinned state is re-applied after. Keeps going on a per-model failure
    (one model refusing to reload must not abandon the rest) and reports each.
    """
    client = _client(request)
    results: list[dict[str, Any]] = []
    for item in body.models:
        name = item.get("model_name")
        if not name:
            continue
        try:
            await client.request(
                "POST", "/api/v1/load", admin=True,
                json_body={"model_name": name}, timeout=CONTROL,
            )
            if item.get("pinned"):
                try:
                    await client.request(
                        "POST", "/internal/pin", admin=True,
                        json_body={"model_name": name, "pinned": True}, timeout=FAST,
                    )
                except UpstreamError as e:
                    logger.warning("Restore re-pin of %s failed: %s", name, e)
            results.append({"model_name": name, "ok": True})
        except UpstreamError as e:
            logger.warning("Restore of %s failed: %s", name, e)
            results.append({"model_name": name, "ok": False,
                            "error_kind": e.kind, "error_detail": str(e)})
    return {"restored": results}


async def _verify_load(
    client: LemonadeClient, model_name: str, sent_body: dict,
) -> tuple[bool | None, str | None]:
    """Confirm the running model reflects the slots/ctx we sent.

    Returns (verified, note). None means "could not verify" (read-back failed) —
    never silently True. Compares ctx_size directly and the slot count via the
    shared F4 parser, so a dropped or altered `--parallel` is caught. `sent_body`
    carries options as TOP-LEVEL fields (Lemonade's load-request shape); /health
    REPORTS them nested under recipe_options, which is what we read back.
    """
    sent = sent_body
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        return None, ("Could not read back the running config, so HELM cannot confirm the "
                      "change took. Re-open Settings to see the current running options.")
    record = next((m for m in (health.get("all_models_loaded") or [])
                   if isinstance(m, dict) and m.get("model_name") == model_name), None)
    if record is None:
        return False, "The model is not loaded after the apply — the change did not take."
    running = record.get("recipe_options") or {}

    mismatches: list[str] = []
    if "ctx_size" in sent and running.get("ctx_size") != sent["ctx_size"]:
        mismatches.append(
            f"total context (asked {sent['ctx_size']:,}, running {running.get('ctx_size')})")
    sent_par = parallel_slots(sent.get("llamacpp_args"))
    run_par = parallel_slots(running.get("llamacpp_args"))
    if sent_par.ok and (not run_par.ok or run_par.value != sent_par.value):
        got = run_par.value if run_par.ok else "none (default 1)"
        mismatches.append(f"slots (asked --parallel {sent_par.value}, running {got})")

    if mismatches:
        return False, (
            "Lemonade reloaded the model but came back running DIFFERENT options than you set — "
            "the change did not fully take: " + "; ".join(mismatches) + ". "
            "Locally-defined models, or Lemonade's own arg re-derivation, can strip settings on "
            "load; setting them via `lemonade load --llamacpp-args=… --save-options` is the "
            "reliable path until this is resolved."
        )
    return True, None


@router.post("/unload", response_model=ActionOut)
async def unload(body: UnloadBody, request: Request, confirm: bool = False) -> ActionOut:
    client = _client(request)
    plan = plan_unload(body.model_name, await _fleet(client))
    out = _out(plan, confirmed=confirm)

    if plan.blocked:
        raise HTTPException(status_code=422, detail=out.model_dump())
    if out.needs_confirmation:
        raise HTTPException(status_code=409, detail=out.model_dump())
    return await _perform(client, "/api/v1/unload", plan.body, out)


class PinBody(BaseModel):
    model_name: str
    pinned: bool


@router.post("/pin", response_model=ActionOut)
async def pin(body: PinBody, request: Request) -> ActionOut:
    """Pin/unpin via the dedicated `POST /internal/pin` route. No reload.

    **Settled by CLI trace, 2026-07-16** (14c §2.1). `lemonade pin`/`unpin`
    both hit `POST /internal/pin` with `{"model_name": ..., "pinned": bool}`,
    and neither reloads the model (the CLI reported success with no "Loading
    model" line). Pin is a metadata flag, not a reprovision.

    This replaces the old load-with-pinned mechanism, which reloaded the model
    for ~10-20s. That cost is gone: toggling pin on the 26B no longer drops it.
    There is no `confirm` gate and no `already_loaded` warning any more, because
    there is nothing disruptive to warn about.

    **Verify by read-back (house rule).** Lemonade returns success on the pin
    call, but HELM does not take its word for it: after the call it re-reads
    `/health` and confirms the `pinned` flag actually flipped. If it did not,
    HELM reports that plainly rather than a false "done" — the pin route body
    shape was reconstructed from a packet capture, and a read-back is what makes
    a subtle mismatch surface as a truthful failure instead of a silent lie.
    """
    client = _client(request)
    health = await _fleet(client)
    loaded = {
        m.get("model_name"): m
        for m in (health.get("all_models_loaded") or [])
        if isinstance(m, dict)
    }
    current = loaded.get(body.model_name)
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{body.model_name} is not loaded, so there is nothing to "
                f"{'pin' if body.pinned else 'unpin'}. Pinning applies to resident models."
            ),
        )

    if current.get("pinned") is body.pinned:
        # Already in the requested state. Report it as done rather than making
        # a no-op call that could look like a failure.
        return ActionOut(
            performed=True,
            plan=PlanOut(blocked=False, needs_confirmation=False, body={}),
            result={"already": True, "pinned": body.pinned},
        )

    plan_out = PlanOut(
        blocked=False, needs_confirmation=False,
        body={"model_name": body.model_name, "pinned": body.pinned},
    )
    try:
        await client.request(
            "POST", "/internal/pin", admin=True,
            json_body={"model_name": body.model_name, "pinned": body.pinned},
            timeout=FAST,  # no reload, so a fast timeout is right
        )
    except UpstreamError as e:
        logger.warning("Pin %s -> %s failed: %s", body.model_name, body.pinned, e)
        return ActionOut(
            performed=False, plan=plan_out, error_kind=e.kind, error_detail=str(e),
        )

    # Read-back. Never accept "it worked" from the call alone.
    if not await _confirm_pin(client, body.model_name, body.pinned):
        return ActionOut(
            performed=False, plan=plan_out,
            error_kind="unconfirmed",
            error_detail=(
                f"Lemonade accepted the pin call but {body.model_name}'s pinned flag did not read "
                f"back as {body.pinned}. Not reporting success for a change HELM cannot confirm. "
                f"Check the model in the dashboard."
            ),
        )
    return ActionOut(performed=True, plan=plan_out, result={"pinned": body.pinned})


async def _confirm_pin(client: LemonadeClient, model_name: str, expected: bool) -> bool:
    """Re-read /health and confirm the pinned flag flipped. Any read failure
    reads as unconfirmed, never as success."""
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        return False
    for model in health.get("all_models_loaded") or []:
        if isinstance(model, dict) and model.get("model_name") == model_name:
            return model.get("pinned") is expected
    return False


async def _perform(
    client: LemonadeClient, path: str, body: dict, out: PlanOut,
) -> ActionOut:
    """Send it. Never collapse the failure modes.

    A rejected key, an NPU-exclusivity refusal, a dead backend and a timeout
    are four different things and four different operator actions (house rule
    #5). `CONTROL` rather than `FAST`: §5.3 says a reload takes 10-20s, and
    reporting a timeout for a load that actually succeeded is the worst lie a
    control plane can tell — the operator retries, and the second load evicts
    something.
    """
    try:
        result = await client.request("POST", path, admin=True, json_body=body, timeout=CONTROL)
    except UpstreamError as e:
        logger.warning("Control action %s failed: %s", path, e)
        return ActionOut(
            performed=False, plan=out, error_kind=e.kind, error_detail=str(e),
        )
    return ActionOut(performed=True, plan=out, result=result)
