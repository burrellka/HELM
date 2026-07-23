"""System/config surface — what HELM knows about itself and its upstream.

This is the one place the frame touches Lemonade, and only to answer "can we
reach it, and can we write to it". It is the model for how every other HELM
router should talk to the upstream: through app.state.lemonade, catching
UpstreamError, and surfacing the KIND — never a generic failure string.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..domain.system_profile import extract_system_profile
from ..upstream.client import FAST, UpstreamError
from ..version import get_version

router = APIRouter(prefix="/api/v1/system", tags=["system"])


class UpstreamStatus(BaseModel):
    reachable: bool
    management_enabled: bool
    lemonade_url: str
    # Populated only on failure. Carries the KIND (unreachable / timeout /
    # unauthorized / ...) and the elapsed time, so the UI can say something
    # true instead of "error". Review F6 / house conflict #5.
    error_kind: str | None = None
    error_detail: str | None = None


class SystemConfig(BaseModel):
    version: str
    auth_enabled: bool
    upstream: UpstreamStatus


@router.get("/config", response_model=SystemConfig)
async def system_config(request: Request) -> SystemConfig:
    """One call the UI can make on load to know what it's dealing with.

    `management_enabled` is what the UI should gate write controls on. Hide
    them when it's false and say why — do not render an Unload button that
    401s. (URSA 1.1.8.)
    """
    client = request.app.state.lemonade

    status = UpstreamStatus(
        reachable=False,
        management_enabled=client.has_admin_key,
        lemonade_url=client._base,  # noqa: SLF001 — read-only display value
    )
    try:
        await client.request("GET", "/live")
        status.reachable = True
    except UpstreamError as e:
        status.reachable = False
        status.error_kind = e.kind
        status.error_detail = str(e)

    return SystemConfig(
        version=get_version(),
        auth_enabled=not bool(getattr(request.app.state, "auth_disabled", False)),
        upstream=status,
    )


@router.get("/profile")
async def system_profile(request: Request) -> dict:
    """The DETECTED hardware of the box HELM is pointed at. Live, not hardcoded.

    This is what makes HELM portable: deploy it to a different Strix Halo (or
    any box) and this reports THAT machine. The Operations "Current system
    setup" panel renders it, and the advisor grounds its hardware reasoning in
    it. Every field is read from the upstream; nothing about a specific machine
    lives in HELM's code.
    """
    client = request.app.state.lemonade

    async def read(path: str):
        try:
            return await client.request("GET", path, timeout=FAST)
        except UpstreamError:
            return None

    info = await read("/api/v1/system-info")
    stats = await read("/api/v1/system-stats")
    health = await read("/api/v1/health")
    params = await read("/api/v1/params")

    profile = extract_system_profile(info, stats, health=health, params=params)
    data = asdict(profile)
    data["backends"] = [asdict(b) for b in profile.backends]
    return data
