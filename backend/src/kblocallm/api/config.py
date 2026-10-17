"""Phase 2: the guarded config surface. Spec §5.3b/§5.3c, 14c Part 2 §2.3.

One pipeline, three doors (raw edit, export, import), two blast-radius levels.
Every mutating path runs the same sequence: read current, validate, diff,
back up the prior version, confirm, apply, and keep a restore point. The
validation lives in `domain.config_edit` (pure, tested); this file does the
I/O and the upstream calls.

**Level A — per-model recipe_options.** Applying reloads ONE model, via the
`POST /api/v1/load` route that already exists. Fully wired.

**Level B — global config.json.** Read and validate are wired (the current
config is readable at `/api/v1/params`). Apply is NOT wired, and deliberately
so: the write mechanism is unconfirmed (14c §2.3 verify item). Either
`lemonade config set` has an HTTP route that must be traced, or config.json is
a host file HELM would edit over a bind-mount it does not yet have. HELM
refuses to fake it, and says exactly which of the two is needed.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..domain.config_edit import (
    ConfigValidation,
    Level,
    validate_global_config,
    validate_recipe_options,
)
from ..storage.config_backups import ConfigBackupStore
from ..upstream.client import CONTROL, FAST, LemonadeClient, UpstreamError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["config"])


class IssueOut(BaseModel):
    level: str
    code: str
    message: str
    key: str | None = None


class ChangeOut(BaseModel):
    key: str
    old: Any = None
    new: Any = None
    kind: str


class ValidationOut(BaseModel):
    blocked: bool
    needs_danger_ack: bool
    dangerous_keys: list[str] = Field(default_factory=list)
    issues: list[IssueOut] = Field(default_factory=list)
    changes: list[ChangeOut] = Field(default_factory=list)


def _validation_out(v: ConfigValidation) -> ValidationOut:
    return ValidationOut(
        blocked=v.blocked,
        needs_danger_ack=v.needs_danger_ack,
        dangerous_keys=v.dangerous_keys,
        issues=[IssueOut(level=i.level.value, code=i.code, message=i.message, key=i.key)
                for i in v.issues],
        changes=[ChangeOut(key=c.key, old=c.old, new=c.new, kind=c.kind) for c in v.changes],
    )


def _client(request: Request) -> LemonadeClient:
    return request.app.state.lemonade


def _require_admin(request: Request) -> LemonadeClient:
    client = _client(request)
    if not client.has_admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HELM has no Lemonade admin key, so config changes are unavailable. "
                "Set HELM_LEMONADE_ADMIN_API_KEY in the stack's .env and restart HELM."
            ),
        )
    return client


# ── Level A: per-model recipe_options ───────────────────────────────────────


async def _model_options(client: LemonadeClient, model_id: str) -> tuple[dict, dict]:
    """Return (running_options, catalog_record) for a model.

    Running from /health (what is in memory), catalog from /models (for the
    trained-ceiling cross-check and recipe). Read-first, always: the editor is
    never handed a blank.
    """
    running: dict = {}
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
        for m in health.get("all_models_loaded") or []:
            if isinstance(m, dict) and m.get("model_name") == model_id:
                running = m.get("recipe_options") or {}
                break
    except UpstreamError:
        pass

    catalog: dict = {}
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
        for m in payload.get("data", []):
            if isinstance(m, dict) and m.get("id") == model_id:
                catalog = m
                break
    except UpstreamError:
        pass
    return running, catalog


@router.get("/model/{model_id:path}")
async def read_model_config(model_id: str, request: Request) -> dict:
    """Read-first (14c §2.3 step 1). The current raw recipe_options as text,
    plus the saved version, so the editor opens on truth, never blank."""
    client = _client(request)
    running, catalog = await _model_options(client, model_id)
    saved = catalog.get("recipe_options") or {}
    return {
        "model_id": model_id,
        "recipe": catalog.get("recipe"),
        "max_context_window": catalog.get("max_context_window"),
        "running_text": json.dumps(running, indent=2),
        "saved_text": json.dumps(saved, indent=2),
    }


class ModelEditBody(BaseModel):
    text: str


@router.post("/model/{model_id:path}/validate", response_model=ValidationOut)
async def validate_model_config(model_id: str, body: ModelEditBody, request: Request) -> ValidationOut:
    """Dry run: validate + diff, no write."""
    client = _client(request)
    running, catalog = await _model_options(client, model_id)
    return _validation_out(validate_recipe_options(
        body.text, current=running, recipe=catalog.get("recipe"),
        max_context_window=catalog.get("max_context_window"),
    ))


@router.post("/model/{model_id:path}/apply")
async def apply_model_config(
    model_id: str, body: ModelEditBody, request: Request, confirm: bool = False,
) -> dict:
    """Apply a raw recipe_options edit. Reloads the model. Full pipeline.

    validate -> refuse if blocked -> require confirm (reload) -> BACK UP the
    prior options -> load with the new options + save_options -> read back.
    """
    client = _require_admin(request)
    running, catalog = await _model_options(client, model_id)
    v = validate_recipe_options(
        body.text, current=running, recipe=catalog.get("recipe"),
        max_context_window=catalog.get("max_context_window"),
    )
    out = _validation_out(v)
    if v.blocked:
        raise HTTPException(status_code=422, detail=out.model_dump())
    if not confirm:
        # Applying reloads the model (~10-20s). Named blast radius before write.
        raise HTTPException(status_code=409, detail={
            "code": "needs_confirmation",
            "message": (
                f"Applying reloads {model_id}: it drops for roughly 10-20 seconds and any "
                f"in-flight request against it fails."
            ),
            "validation": out.model_dump(),
        })

    # Back up the PRIOR options before writing (14c §2.3 step 5).
    backup_id = ConfigBackupStore(request.app.state.db).save(
        f"model:{model_id}", json.dumps(running, indent=2), reason="apply")

    # /api/v1/load takes recipe options as TOP-LEVEL fields (ctx_size,
    # llamacpp_args, …), not nested under `recipe_options` — that is the /health
    # RESPONSE shape, not the request shape. Spread the edited options onto the
    # body, or Lemonade ignores them and the edit silently no-ops.
    load_body: dict[str, Any] = {
        "model_name": model_id, "save_options": True, **v.parsed,
    }
    if catalog.get("recipe"):
        load_body["recipe"] = catalog["recipe"]
    try:
        await client.request("POST", "/api/v1/load", admin=True,
                             json_body=load_body, timeout=CONTROL)
    except UpstreamError as e:
        return {"applied": False, "backup_id": backup_id,
                "error_kind": e.kind, "error_detail": str(e)}

    verified = await _readback_options(client, model_id, v.parsed)
    return {"applied": True, "backup_id": backup_id, "verified": verified}


@router.post("/model/{model_id:path}/reset")
async def reset_model_config(model_id: str, request: Request, confirm: bool = False) -> dict:
    """Reset-to-auto: clear persisted overrides, reload with GGUF auto-detect.

    **Provisional mechanism (14c §2.2, architect-flagged).** There is no reset
    CLI verb, so HELM reloads with `save_options: true` and NO recipe_options,
    which should persist "no overrides" and let auto-detection size the model.
    Whether Lemonade treats that as a genuine clear is not yet confirmed on the
    box, so this reads back the result and reports honestly: if the saved
    overrides did not clear, it says so rather than claiming a reset that did
    not happen. This is the V3.1 stale-CONTEXT_SIZE lesson (a stale manual
    context crashed the backend; auto-detect is the safe default).
    """
    client = _require_admin(request)
    running, catalog = await _model_options(client, model_id)
    if not confirm:
        raise HTTPException(status_code=409, detail={
            "code": "needs_confirmation",
            "message": (
                f"Reset reloads {model_id} with auto-detected defaults, discarding its saved "
                f"slots and context. It drops for roughly 10-20 seconds."
            ),
        })

    backup_id = ConfigBackupStore(request.app.state.db).save(
        f"model:{model_id}", json.dumps(running, indent=2), reason="reset")

    body: dict[str, Any] = {"model_name": model_id, "save_options": True}
    if catalog.get("recipe"):
        body["recipe"] = catalog["recipe"]
    try:
        await client.request("POST", "/api/v1/load", admin=True, json_body=body, timeout=CONTROL)
    except UpstreamError as e:
        return {"applied": False, "backup_id": backup_id,
                "error_kind": e.kind, "error_detail": str(e)}

    # Read back the SAVED options. If they still carry manual overrides, the
    # reset mechanism did not clear them, and we must not pretend it did.
    _, catalog_after = await _model_options(client, model_id)
    saved_after = catalog_after.get("recipe_options") or {}
    cleared = not saved_after.get("ctx_size") and not saved_after.get("llamacpp_args")
    return {
        "applied": True, "backup_id": backup_id, "cleared": cleared,
        "saved_after": saved_after,
        "note": None if cleared else (
            "Reload succeeded but the saved overrides did not clear. The reset mechanism on this "
            "Lemonade build may differ; the model's saved options are unchanged. Flagging for the "
            "architect rather than reporting a reset that did not take."
        ),
    }


async def _readback_options(client: LemonadeClient, model_id: str, sent: dict) -> bool:
    """Confirm the running model now reflects the ctx_size we sent."""
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        return False
    for m in health.get("all_models_loaded") or []:
        if isinstance(m, dict) and m.get("model_name") == model_id:
            now = m.get("recipe_options") or {}
            if "ctx_size" in sent:
                return now.get("ctx_size") == sent["ctx_size"]
            return True
    return False


@router.get("/model/{model_id:path}/export")
async def export_model_config(model_id: str, request: Request) -> dict:
    """§5.3c: snapshot a model's saved options for backup or replication."""
    _, catalog = await _model_options(_client(request), model_id)
    return {"model_id": model_id, "recipe": catalog.get("recipe"),
            "recipe_options": catalog.get("recipe_options") or {}}


# ── Level B: global config.json ─────────────────────────────────────────────


@router.get("/global")
async def read_global_config(request: Request) -> dict:
    """Read-first. The live config.json, from /internal/config.

    The old unauth /api/v1/params was removed in Lemonade 11.8; its replacement
    /internal/config is admin-key-gated, so this screen now needs the key.
    """
    client = _client(request)
    try:
        params = await client.request("GET", "/internal/config", admin=True, timeout=FAST)
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=f"{e.kind}: {e}")
    # Tolerate a {"config": {...}} wrapper in case the new endpoint nests it.
    if isinstance(params.get("config"), dict):
        params = params["config"]
    # Drop the generator banner if present; it is not a settable key.
    params = {k: v for k, v in params.items() if k != "_generated"}
    return {"text": json.dumps(params, indent=2), "current": params}


class GlobalEditBody(BaseModel):
    text: str
    current: dict[str, Any] | None = None


@router.post("/global/validate", response_model=ValidationOut)
async def validate_global(body: GlobalEditBody, request: Request) -> ValidationOut:
    current = body.current
    if current is None:
        try:
            current = await _client(request).request(
                "GET", "/internal/config", admin=True, timeout=FAST)
            if isinstance(current.get("config"), dict):
                current = current["config"]
            current = {k: v for k, v in current.items() if k != "_generated"}
        except UpstreamError:
            current = {}
    return _validation_out(validate_global_config(body.text, current=current))


@router.post("/global/apply")
async def apply_global(body: GlobalEditBody, request: Request) -> dict:
    """NOT wired, on purpose. The write mechanism is unconfirmed (14c §2.3).

    HELM validates and diffs a global edit, but it will not apply one until the
    box answers one question: is config.json writable via an API route (trace
    `lemonade config set`), or is it a host file HELM must edit over a
    bind-mount it does not yet have? Faking an apply here is the exact
    foot-gun the pipeline exists to prevent. So it refuses, loudly, with the
    two ways to unblock it.
    """
    _require_admin(request)  # still gate it, so the 501 is not a bypass
    raise HTTPException(status_code=501, detail={
        "code": "apply_not_wired",
        "message": (
            "HELM can read, validate and diff the global config, but applying a change to "
            "config.json is not yet wired, because the write mechanism is unconfirmed. Two ways "
            "to unblock it: (1) trace `lemonade config set` to find its HTTP route, or (2) "
            "bind-mount config.json into the HELM container so it can edit the file directly. "
            "A global change is assumed to require a FULL server restart that drops every model, "
            "so HELM will not guess at it."
        ),
    })


# ── backups / restore (shared, both levels) ─────────────────────────────────


@router.get("/backups")
def list_backups(request: Request, scope: str) -> dict:
    """Restore history for a scope ('model:<id>' or 'global')."""
    store = ConfigBackupStore(request.app.state.db)
    return {"backups": [
        {"id": b.id, "reason": b.reason, "created_at": b.created_at,
         "preview": b.content[:200]}
        for b in store.list(scope)
    ]}


@router.post("/backups/{backup_id}/restore")
async def restore_backup(backup_id: int, request: Request, confirm: bool = False) -> dict:
    """One-click restore (14c §2.3 step 7). Replays a backup through apply.

    Only Level A (model) restores are wired, because Level B apply is not. A
    global restore returns the same not-wired refusal as a global apply, rather
    than pretending the rail exists where the write does not.
    """
    client = _require_admin(request)
    backup = ConfigBackupStore(request.app.state.db).get(backup_id)
    if backup is None:
        raise HTTPException(status_code=404, detail=f"No backup {backup_id}.")

    if backup.scope == "global":
        raise HTTPException(status_code=501, detail={
            "code": "apply_not_wired",
            "message": "Global config apply is not wired yet, so a global restore cannot run. "
                       "See the global apply endpoint for how to unblock it.",
        })

    model_id = backup.scope.removeprefix("model:")
    if not confirm:
        raise HTTPException(status_code=409, detail={
            "code": "needs_confirmation",
            "message": f"Restoring reloads {model_id} with the backed-up options (~10-20s drop).",
        })

    catalog_recipe = None
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
        for m in payload.get("data", []):
            if isinstance(m, dict) and m.get("id") == model_id:
                catalog_recipe = m.get("recipe")
                break
    except UpstreamError:
        pass

    # Top-level fields, not nested under recipe_options (Lemonade's load request
    # shape — see apply_model_config above).
    load_body: dict[str, Any] = {
        "model_name": model_id,
        "save_options": True,
        **json.loads(backup.content),
    }
    if catalog_recipe:
        load_body["recipe"] = catalog_recipe
    try:
        await client.request("POST", "/api/v1/load", admin=True,
                             json_body=load_body, timeout=CONTROL)
    except UpstreamError as e:
        return {"restored": False, "error_kind": e.kind, "error_detail": str(e)}
    return {"restored": True, "model_id": model_id}
