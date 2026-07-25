"""Per-model live detail (14c §1.12). Read-only.

Right now this hosts one endpoint — the running-context drift read, landed as
its own small §6 correctness change ahead of the full normalized detail
endpoint the architect blessed. The routing namespace (`/api/v1/models/{id}/…`)
is his; the detail endpoint will join it here.

The drift read is deliberately NOT folded into the dashboard poll. `/slots` is
an extra upstream round trip, and §1.12's own guidance is to keep it off the
always-on 5s cadence — it is fetched only when the operator opens a model's
detail. A slow or failed `/slots` degrades to `unknown`, never to a number.
"""
from __future__ import annotations

import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from ..domain.concurrency import SLOTTED_RECIPES
from ..domain.ctx_drift import check_ctx_drift
from ..upstream.client import FAST, LemonadeClient, UpstreamError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class CtxDriftOut(BaseModel):
    """Tri-state, by construction. `state` is match | drift | unknown; the two
    numbers are present only when we can stand behind them, and `detail` always
    explains — especially the unknowns, which must not read as reassurance."""

    state: str
    computed_per_slot: int | None = None
    running_n_ctx: int | None = None
    detail: str


class CtxDriftRow(CtxDriftOut):
    """One model's drift in the batch payload."""

    model_name: str


class CtxDriftBatch(BaseModel):
    results: list[CtxDriftRow]


def _out(result) -> dict:
    return {
        "state": result.state.value,
        "computed_per_slot": result.computed_per_slot,
        "running_n_ctx": result.running_n_ctx,
        "detail": result.detail,
    }


async def _read_health(client: LemonadeClient) -> list[dict]:
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not read the fleet from Lemonade ({e.kind}): {e}",
        ) from e
    loaded = health.get("all_models_loaded") if isinstance(health, dict) else None
    return loaded if isinstance(loaded, list) else []


async def _read_slots(client: LemonadeClient, why: str):
    """One /slots read. A failure is `unknown`, never no-drift, so it returns
    None and the domain degrades. Off the dashboard poll — see module docstring."""
    try:
        return await client.request("GET", "/api/v1/slots", timeout=FAST)
    except UpstreamError as e:
        logger.info("ctx-check: /slots read failed (%s): %s", why, e)
        return None


@router.get("/ctx-drift", response_model=CtxDriftBatch)
async def ctx_drift_all(request: Request) -> CtxDriftBatch:
    """Running-context drift for EVERY loaded llamacpp model, in ONE /slots read.

    This is the dashboard's shape: the operator sees at a glance which model's
    running context is attributable (and matches or drifts) and which read
    `unknown` because Lemonade's model-blind /slots can't be pinned to them. One
    upstream /slots call, shared across all rows — deliberately NOT part of the
    5s dashboard poll (§1.12: keep /slots off the always-on cadence).
    """
    client: LemonadeClient = request.app.state.lemonade
    loaded = await _read_health(client)
    loaded_llamacpp = [m for m in loaded if m.get("recipe") in SLOTTED_RECIPES]

    # Skip the /slots round trip entirely when nothing could use it.
    slots_payload = await _read_slots(client, "batch") if loaded_llamacpp else None

    rows = [
        CtxDriftRow(model_name=m.get("model_name") or "",
                    **_out(check_ctx_drift(m, loaded_llamacpp, slots_payload)))
        for m in loaded_llamacpp
    ]
    return CtxDriftBatch(results=rows)


@router.get("/{model_id:path}/ctx-check", response_model=CtxDriftOut)
async def ctx_check(model_id: str, request: Request) -> CtxDriftOut:
    """Does the context this model is ACTUALLY running match what HELM computed?

    Reads /health for the loaded fleet and /slots for the running instance, then
    attributes /slots to a model by its slot count (Lemonade's /slots carries no
    model label). Unattributable -> unknown. See `domain.ctx_drift`.
    """
    model_id = unquote(model_id)
    client: LemonadeClient = request.app.state.lemonade

    loaded = await _read_health(client)
    record = next((m for m in loaded if m.get("model_name") == model_id), None)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model_id!r} is not currently loaded, so it has no running slots.",
        )

    loaded_llamacpp = [m for m in loaded if m.get("recipe") in SLOTTED_RECIPES]

    # /slots only when it can matter: a non-llamacpp model has no slots to read,
    # so skip the round trip entirely and let the domain return its unknown.
    slots_payload = None
    if record.get("recipe") in SLOTTED_RECIPES:
        slots_payload = await _read_slots(client, model_id)

    return CtxDriftOut(**_out(check_ctx_drift(record, loaded_llamacpp, slots_payload)))
