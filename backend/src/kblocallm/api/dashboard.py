"""Phase 1: the read-only dashboard. Handoff §2.

Five upstream reads, fetched concurrently, assembled by `domain.dashboard`
into one payload. No writes exist in this module and none should: Phase 1 is
read-only, and the router is mounted with `_AUTH_REQUIRED` regardless (F1a —
auth holds by construction, so the Phase 2 control endpoints cannot land here
unguarded).

**Cadence.** The spec's warning is that the dashboard must not become a load
source on the box it watches. So:
  - `/live` at 2s — the router does no work for it (spec §4.1).
  - `/dashboard` at ~5s — health + system-stats + stats.
  - system-info and params are near-static, cached here, and refreshed rarely.
    They are the two heaviest reads: system-info alone is ~1,200 lines of
    recipe/backend metadata, and re-pulling it every 5s to re-read one
    unchanging pool figure would be silly.

**Concurrency.** The reads are independent, so they run in a task group rather
than serially; four sequential round trips would make the poll interval
partly a function of our own latency.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..config import get_settings
from ..domain.dashboard import Dashboard, SectionError, build_dashboard
from ..domain.gpu_sensors import GpuSensors, read_gpu_sensors
from ..storage.backends import BackendStateStore
from ..upstream.client import FAST, LemonadeClient, UpstreamError, UpstreamErrorKind

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

#: system-info and params describe hardware and server config. They change
#: when someone edits config and restarts lemond, not between polls. Five
#: minutes keeps a config change visible within one coffee refill while taking
#: the big read off the hot path.
STATIC_TTL_SECONDS = 300.0


class Liveness(BaseModel):
    """The 2s poll. Deliberately tiny.

    Separate from /dashboard so the UI can distinguish "the box is up but a
    read failed" from "the box is gone" — those need different operator
    reactions, and a single combined endpoint would collapse them.
    """

    alive: bool
    #: Round-trip in ms. A /live that answers in 900ms is a box in trouble,
    #: and that is worth seeing BEFORE it starts failing.
    latency_ms: int | None = None
    error_kind: str | None = None
    error_detail: str | None = None


class _StaticCache:
    """TTL cache for the two near-static reads.

    Holds the last good value past its TTL on failure, and reports how stale
    it is. Serving a five-minute-old pool figure while the box is briefly
    unreachable is right; blanking the memory bar because one refresh missed
    is not — the number did not change, we just failed to re-read it.
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._fetched_at: dict[str, float] = {}
        self._errors: dict[str, SectionError] = {}
        self._lock = asyncio.Lock()

    def is_fresh(self, key: str) -> bool:
        at = self._fetched_at.get(key)
        return at is not None and (time.monotonic() - at) < STATIC_TTL_SECONDS

    async def get(
        self, key: str, fetch, force: bool = False,
    ) -> tuple[Any | None, SectionError | None]:
        async with self._lock:
            if not force and self.is_fresh(key):
                return self._values.get(key), self._errors.get(key)
            try:
                value = await fetch()
            except UpstreamError as e:
                error = SectionError(kind=e.kind, detail=str(e))
                self._errors[key] = error
                # Keep the stale value: a pool that we read successfully five
                # minutes ago is still the pool.
                return self._values.get(key), error
            self._values[key] = value
            self._fetched_at[key] = time.monotonic()
            self._errors.pop(key, None)
            return value, None

    def invalidate(self) -> None:
        self._fetched_at.clear()


def _cache(request: Request) -> _StaticCache:
    cache = getattr(request.app.state, "static_cache", None)
    if cache is None:
        cache = _StaticCache()
        request.app.state.static_cache = cache
    return cache


@router.get("/live", response_model=Liveness)
async def live(request: Request) -> Liveness:
    """Liveness. The 2s poll — /live does no work upstream (spec §4.1)."""
    client: LemonadeClient = request.app.state.lemonade
    started = time.monotonic()
    try:
        await client.request("GET", "/live", timeout=FAST)
    except UpstreamError as e:
        return Liveness(alive=False, error_kind=e.kind, error_detail=str(e))
    return Liveness(alive=True, latency_ms=int((time.monotonic() - started) * 1000))


@router.get("/dashboard", response_model=Dashboard)
async def dashboard(request: Request, refresh_static: bool = False) -> Dashboard:
    """The Phase 1 payload. Always 200; failures live in `errors`.

    A partial dashboard is the point. If /stats times out, the models table
    and the memory bar are still true and still worth showing — and the
    operator is usually here BECAUSE something is wrong, which is the worst
    possible moment to blank the screen over one bad read out of five.
    """
    client: LemonadeClient = request.app.state.lemonade
    cache = _cache(request)
    errors: dict[str, SectionError] = {}

    async def read(path: str):
        return await client.request("GET", path, timeout=FAST)

    async def read_admin(path: str):
        # /internal/* is key-gated (the old unauth /api/v1/params was removed in
        # Lemonade 11.8). Keyless/read-only installs get NO_ADMIN_KEY here, which
        # the GTT cross-check degrades to "unconfirmed" — the pool number itself
        # comes from system-info/system-stats and needs no key.
        return await client.request("GET", path, admin=True, timeout=FAST)

    # The three live reads, concurrently. Independent calls; serial round
    # trips would make our own latency part of the poll interval.
    live_results = await asyncio.gather(
        read("/api/v1/health"),
        read("/api/v1/system-stats"),
        read("/api/v1/stats"),
        return_exceptions=True,
    )
    health, system_stats, stats = (
        _unwrap(name, result, errors)
        for name, result in zip(("health", "system_stats", "stats"), live_results)
    )

    system_info, info_error = await cache.get(
        "system_info", lambda: read("/api/v1/system-info"), force=refresh_static,
    )
    if info_error:
        errors["system_info"] = info_error

    params, params_error = await cache.get(
        "config", lambda: read_admin("/internal/config"), force=refresh_static,
    )
    if params_error:
        errors["config"] = params_error

    backend_states = {}
    if isinstance(health, dict):
        models = health.get("all_models_loaded")
        if isinstance(models, list):
            try:
                store = BackendStateStore(request.app.state.db)
                # Sync SQLite in an async handler: offload so a slow disk on
                # the bind mount cannot stall the event loop and, with it,
                # every other poll.
                backend_states = await asyncio.to_thread(store.observe_all, models)
            except Exception as e:  # noqa: BLE001
                # Crash detection is an enrichment. If HELM's own DB is
                # broken, that is worth a loud log and a visible section
                # error, but the models table is still true and still ships.
                logger.exception("Backend state tracking failed")
                errors["crash_detection"] = SectionError(
                    kind="storage_error",
                    detail=(
                        f"HELM could not read its own pid history, so restart "
                        f"detection is unavailable this poll: {e}"
                    ),
                )

    # Local GPU thermals — temp/power the kernel gets right even where the
    # utilization gauge is stuck. A tiny sysfs read, but offloaded like the
    # SQLite call so a blocking read on an odd /sys can't stall the event loop.
    # Absent (HELM not on the GPU host, non-Linux dev box) -> all-None, no fake.
    try:
        sensors = await asyncio.to_thread(read_gpu_sensors, get_settings().drm_root)
    except Exception:  # noqa: BLE001
        logger.debug("GPU sensor read failed", exc_info=True)
        sensors = GpuSensors()

    return build_dashboard(
        health=health,
        system_stats=system_stats,
        system_info=system_info,
        stats=stats,
        params=params,
        backend_states=backend_states,
        gpu_temp_c=sensors.temp_c,
        gpu_power_w=sensors.power_w,
        reachable="health" not in errors,
        management_enabled=client.has_admin_key,
        errors=errors,
    )


def _unwrap(
    section: str,
    result: Any,
    errors: dict[str, SectionError],
) -> dict | None:
    """Turn one gathered result into a payload or a recorded section error.

    Keeps the UpstreamError KIND — "unreachable" and "unauthorized" and
    "timeout" are three different operator actions and must not collapse into
    "error" (house rule #5).
    """
    if isinstance(result, UpstreamError):
        errors[section] = SectionError(kind=result.kind, detail=str(result))
        return None
    if isinstance(result, BaseException):
        logger.exception("Unexpected failure reading %s", section, exc_info=result)
        errors[section] = SectionError(
            kind=UpstreamErrorKind.UPSTREAM_ERROR,
            detail=f"Unexpected error reading {section}: {result}",
        )
        return None
    return result
