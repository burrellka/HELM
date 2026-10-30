"""The Phase 1 view model. Handoff §2.

Assembles /health + /system-stats + /system-info + /stats + /params into the
thing the dashboard renders. Every derived number comes from a tested module
in this package (`pool`, `concurrency`, `crash`); this file's job is assembly,
per-section honesty, and nothing else.

**Why the derivation is here and not in TypeScript.** The correctness rules are
the product. They need a table test, they need to be one implementation, and
they need to survive someone editing a React component at 11pm. Python has the
fixtures and the tests, so the frontend receives decided values and a `display`
string, never a raw `ctx_size` it might divide itself.

**Pydantic, not dataclasses, deliberately.** These models ARE the wire contract
with the SPA, so one definition serializes and documents itself. Pydantic is a
data library; the no-FastAPI rule in this package still holds.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .concurrency import ConcurrencyKind, derive_concurrency
from .crash import BackendState, BackendStatus
from .golden import GoldenConfig, GoldenState, compare_golden
from .gpu_sensors import interpret_gpu_gauge
from .pool import MemoryPool, PoolWarning, extract_pool

#: Above this, a reported tok/s is an artifact, not a measurement. The live box
#: returns exactly 1000000.0 after a 1-token response — a divide by a
#: near-zero elapsed. The fastest thing this hardware does is low thousands, so
#: anything past 10k is the artifact. We show the reported number AND say it is
#: implausible; we do not silently pass "1,000,000 tok/s" off as throughput.
PLAUSIBLE_MAX_TOKENS_PER_SECOND = 10_000.0


class SectionError(BaseModel):
    """One upstream call's failure, kept local to its section.

    The point is partial degradation: if /stats times out, the models table
    and the memory bar are still true and still worth showing. Blanking the
    whole dashboard because one of four calls failed throws away the reads
    that DID work — and the operator is usually looking at HELM precisely
    because something is wrong.
    """

    kind: str
    detail: str


class PoolView(BaseModel):
    capacity_gb: float | None = None
    used_gb: float | None = None
    percent: float | None = None
    #: The field path this came from, e.g.
    #: "system-info.devices.amd_gpu[0].virtual_mem_gb". On screen, so the next
    #: person asking "where does 120 come from?" reads the answer instead of
    #: excavating for it.
    source: str | None = None
    warning: str | None = None
    warning_detail: str | None = None
    gtt_enabled: bool | None = None


class TypeCapacity(BaseModel):
    """Rule 6: max_models is PER TYPE. Never render 3/10 as a global cap."""

    type: str
    loaded: int
    max: int | None
    pinned: int


class ResourceView(BaseModel):
    pool: PoolView
    gpu_percent: float | None = None
    #: GPU temperature (°C) and average power (W), read locally from hwmon —
    #: Lemonade exposes neither. None when HELM is not co-located with the GPU.
    #: These read correctly even when gpu_percent is the stuck gauge below.
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    #: Set when gpu_percent is provably lying: pinned at 100% while temp/power
    #: say idle (the gfx1151 amdgpu bug). The gauge still renders; this note
    #: sits beside it. Self-clears if the driver is ever fixed upstream.
    gpu_percent_note: str | None = None
    cpu_percent: float | None = None
    #: Rule 3. ALWAYS null on this box — a known-broken gauge. The frontend
    #: renders `?? 'unavailable'`, never `|| `: if the gauge is ever fixed
    #: upstream and legitimately reports 0, `||` would render a true 0% as
    #: "unavailable", replacing the old lie with a new one.
    npu_percent: float | None = None
    #: system-stats.memory_gb — host RAM in use. NOT the GPU pool, and not a
    #: denominator for anything here.
    system_ram_gb: float | None = None
    capacity: list[TypeCapacity] = Field(default_factory=list)


class ConcurrencyView(BaseModel):
    kind: str
    display: str
    slots: int | None = None
    per_slot_ctx: int | None = None
    slots_defaulted: bool = False
    slot_policy: str | None = None
    note: str | None = None


class GoldenDriftItemView(BaseModel):
    key: str
    golden: str | None = None
    running: str | None = None
    critical: bool = False


class GoldenDriftView(BaseModel):
    #: "match" | "drift" | "unknown".
    state: str
    items: list[GoldenDriftItemView] = Field(default_factory=list)
    has_critical: bool = False


class ModelRow(BaseModel):
    model_name: str
    checkpoint: str | None = None
    device: str | None = None
    type: str | None = None
    recipe: str | None = None
    llamacpp_backend: str | None = None
    #: "llamacpp/rocm", "flm". Assembled here so the column is one tested
    #: string rather than two fields the frontend concatenates.
    backend_label: str
    concurrency: ConcurrencyView
    pinned: bool = False
    max_context_window: int | None = None

    status: str
    raw_status: str | None = None
    status_detail: str | None = None
    restarted_now: bool = False
    last_restart_at: datetime | None = None
    restart_is_recent: bool = False
    watchdog_reset: bool = False

    #: The raw LRU clock. Monotonic, unknown unit, no wall-clock anchor
    #: anywhere in the API — so it is NOT rendered as "3m ago". See
    #: `lru_rank`.
    last_use: int | None = None
    #: 1 = most recently used. This is what the LRU column is actually FOR:
    #: under capacity pressure the router evicts the least recently used
    #: unpinned model, and rank answers that question exactly, without
    #: inventing a duration out of a counter whose unit nobody has confirmed.
    lru_rank: int | None = None
    #: True when this is the unpinned model the router would evict first.
    next_to_evict: bool = False

    pid: int | None = None
    port: int | None = None
    backend_url: str | None = None

    #: Drift of the running config from the operator's golden baseline, when one
    #: is pinned for this model. None = no golden pinned (the common case). This
    #: is how a silently-dropped `--reasoning off` becomes a visible alarm.
    golden_drift: GoldenDriftView | None = None


class ThroughputView(BaseModel):
    tokens_per_second: float | None = None
    time_to_first_token: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_count_total: int | None = None
    input_tokens_total: int | None = None
    output_tokens_total: int | None = None
    #: True when tokens_per_second is an artifact rather than a measurement.
    #: The UI must show the honest-ignorance treatment, not the number.
    rate_implausible: bool = False
    rate_note: str | None = None


class Dashboard(BaseModel):
    generated_at: datetime
    lemonade_version: str | None = None
    #: False when /live did not answer. The table may still hold the last good
    #: read; this says whether to trust it as current.
    reachable: bool = True
    management_enabled: bool = False

    models: list[ModelRow] = Field(default_factory=list)
    resources: ResourceView
    throughput: ThroughputView | None = None

    #: Per-section failures. Keyed by section: health / system_stats /
    #: system_info / stats / params.
    errors: dict[str, SectionError] = Field(default_factory=dict)


# ---------------------------------------------------------------------------


def _pool_view(pool: MemoryPool) -> PoolView:
    return PoolView(
        capacity_gb=pool.capacity_gb,
        used_gb=pool.used_gb,
        percent=pool.percent,
        source=pool.source,
        warning=pool.warning.value if isinstance(pool.warning, PoolWarning) else None,
        warning_detail=pool.warning_detail,
        gtt_enabled=pool.gtt_enabled,
    )


def _golden_drift(golden: GoldenConfig | None, model: dict[str, Any]) -> GoldenDriftView | None:
    """Compare a model's running config to its golden, if one is pinned. None
    when no golden exists (the common case) and when it MATCHES — the row only
    carries a golden_drift object when there is something to say (drift or an
    unverifiable read)."""
    if golden is None:
        return None
    drift = compare_golden(golden, model)
    if drift.state is GoldenState.MATCH:
        return None
    return GoldenDriftView(
        state=drift.state.value,
        items=[GoldenDriftItemView(key=i.key, golden=i.golden, running=i.running,
                                   critical=i.critical) for i in drift.items],
        has_critical=drift.has_critical,
    )


def _number(value: Any) -> float | None:
    """A float, or None. Never a coerced zero.

    `or 0.0` is how a null gauge becomes a confident 0% — the exact shape of
    Rule 3's complaint.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _capacity(health: dict[str, Any], models: list[dict]) -> list[TypeCapacity]:
    """Rule 6. Loaded counts per type against the per-type ceiling.

    Union of the types the router publishes limits for and the types actually
    loaded: a loaded type with no published limit still needs a row (its count
    is real), and a limit with nothing loaded is still worth showing as free
    capacity.
    """
    max_models = health.get("max_models")
    max_models = max_models if isinstance(max_models, dict) else {}
    pinned = health.get("pinned_models")
    pinned = pinned if isinstance(pinned, dict) else {}

    loaded_counts: dict[str, int] = {}
    for model in models:
        loaded_counts[model.get("type") or "unknown"] = (
            loaded_counts.get(model.get("type") or "unknown", 0) + 1
        )

    rows = [
        TypeCapacity(
            type=type_name,
            loaded=loaded_counts.get(type_name, 0),
            max=max_models.get(type_name) if isinstance(max_models.get(type_name), int) else None,
            pinned=pinned.get(type_name, 0) if isinstance(pinned.get(type_name), int) else 0,
        )
        for type_name in sorted(set(max_models) | set(loaded_counts))
    ]
    # Loaded types first: the operator cares about what is resident before
    # what is theoretically permitted.
    rows.sort(key=lambda r: (r.loaded == 0, r.type))
    return rows


def _backend_label(model: dict[str, Any]) -> str:
    """'llamacpp/rocm' or 'flm'. Never a bare slash with nothing after it."""
    recipe = model.get("recipe") or "unknown"
    options = model.get("recipe_options") or {}
    backend = options.get("llamacpp_backend")
    return f"{recipe}/{backend}" if backend else recipe


def _port(model: dict[str, Any]) -> int | None:
    """The subprocess port out of backend_url (http://127.0.0.1:8001/v1)."""
    url = model.get("backend_url")
    if not isinstance(url, str) or ":" not in url:
        return None
    try:
        return int(url.rsplit(":", 1)[1].split("/")[0])
    except (ValueError, IndexError):
        return None


def _lru_ranks(models: list[dict[str, Any]]) -> dict[int, int]:
    """Rank models by last_use. 1 = most recent. Keyed by index.

    Ordering is all a monotonic counter honestly supports: it has no epoch and
    the API exposes no current value to compare against, so "3m ago" would be
    fabricated. Ordering is also the question the LRU column exists to answer
    — the router evicts the least recently used unpinned model, and rank says
    which that is.

    Models with no usable last_use get no rank rather than sorting to the end
    as if they were the oldest.
    """
    usable = [
        (i, m["last_use"]) for i, m in enumerate(models)
        if isinstance(m.get("last_use"), int) and not isinstance(m.get("last_use"), bool)
    ]
    usable.sort(key=lambda pair: pair[1], reverse=True)
    return {index: rank for rank, (index, _) in enumerate(usable, start=1)}


def build_dashboard(
    health: dict[str, Any] | None,
    system_stats: dict[str, Any] | None,
    system_info: dict[str, Any] | None,
    stats: dict[str, Any] | None,
    params: dict[str, Any] | None,
    backend_states: dict[str, BackendState] | None = None,
    *,
    golden_configs: dict[str, GoldenConfig] | None = None,
    gpu_temp_c: float | None = None,
    gpu_power_w: float | None = None,
    reachable: bool = True,
    management_enabled: bool = False,
    errors: dict[str, SectionError] | None = None,
    now: datetime | None = None,
) -> Dashboard:
    """Assemble the dashboard. Any payload may be None; the section degrades.

    `backend_states` comes from the BackendStateStore (it needs the DB, which
    is why it is passed in rather than computed here).
    """
    now = now or datetime.now(timezone.utc)
    health = health if isinstance(health, dict) else {}
    params = params if isinstance(params, dict) else {}
    backend_states = backend_states or {}
    golden_configs = golden_configs or {}

    raw_models = health.get("all_models_loaded")
    raw_models = raw_models if isinstance(raw_models, list) else []
    # Filter to dicts ONCE, here, rather than defensively in each helper. A
    # single malformed entry must cost that row and nothing else — every other
    # model in the table is still true, and the operator is usually reading
    # this because something is already wrong.
    raw_models = [m for m in raw_models if isinstance(m, dict)]
    recipes = (system_info or {}).get("recipes") if isinstance(system_info, dict) else None

    # `enable_dgpu_gtt` now comes from /internal/config (the old /api/v1/params
    # was removed). Read it flat, and tolerate a one-level {"config": {...}}
    # wrapper in case the new endpoint nests it — either way a non-bool or a
    # missing key degrades to None ("GTT unconfirmed"), never a guessed bool.
    gtt = params.get("enable_dgpu_gtt")
    if not isinstance(gtt, bool):
        nested = params.get("config")
        if isinstance(nested, dict):
            gtt = nested.get("enable_dgpu_gtt")
    gtt_enabled = gtt if isinstance(gtt, bool) else None

    ranks = _lru_ranks(raw_models)
    evict_index = _next_to_evict(raw_models, ranks)

    rows: list[ModelRow] = []
    for index, model in enumerate(raw_models):
        name = model.get("model_name") or "(unnamed)"
        concurrency = derive_concurrency(model, recipes)
        state = backend_states.get(name)
        rows.append(
            ModelRow(
                model_name=name,
                checkpoint=model.get("checkpoint"),
                device=model.get("device"),
                type=model.get("type"),
                recipe=model.get("recipe"),
                llamacpp_backend=(model.get("recipe_options") or {}).get("llamacpp_backend"),
                backend_label=_backend_label(model),
                concurrency=ConcurrencyView(
                    kind=concurrency.kind.value
                    if isinstance(concurrency.kind, ConcurrencyKind)
                    else str(concurrency.kind),
                    display=concurrency.display,
                    slots=concurrency.slots,
                    per_slot_ctx=concurrency.per_slot_ctx,
                    slots_defaulted=concurrency.slots_defaulted,
                    slot_policy=concurrency.slot_policy,
                    note=concurrency.note,
                ),
                pinned=model.get("pinned") is True,
                max_context_window=model.get("max_context_window"),
                status=_status_value(state, model),
                raw_status=model.get("status"),
                status_detail=state.detail if state else None,
                restarted_now=state.restarted_now if state else False,
                last_restart_at=state.last_restart_at if state else None,
                restart_is_recent=state.is_recent_restart(now) if state else False,
                watchdog_reset=model.get("watchdog_reset") is True,
                last_use=model.get("last_use") if isinstance(model.get("last_use"), int) else None,
                lru_rank=ranks.get(index),
                next_to_evict=index == evict_index,
                pid=model.get("pid"),
                port=_port(model),
                backend_url=model.get("backend_url"),
                golden_drift=_golden_drift(golden_configs.get(name), model),
            )
        )

    pool = extract_pool(system_info, system_stats, gtt_enabled=gtt_enabled)
    stats_dict = system_stats if isinstance(system_stats, dict) else {}
    gpu_percent = _number(stats_dict.get("gpu_percent"))
    resources = ResourceView(
        pool=_pool_view(pool),
        gpu_percent=gpu_percent,
        gpu_temp_c=gpu_temp_c,
        gpu_power_w=gpu_power_w,
        gpu_percent_note=interpret_gpu_gauge(gpu_percent, gpu_temp_c, gpu_power_w),
        cpu_percent=_number(stats_dict.get("cpu_percent")),
        npu_percent=_number(stats_dict.get("npu_percent")),
        system_ram_gb=_number(stats_dict.get("memory_gb")),
        capacity=_capacity(health, raw_models),
    )

    return Dashboard(
        generated_at=now,
        lemonade_version=health.get("version"),
        reachable=reachable,
        management_enabled=management_enabled,
        models=rows,
        resources=resources,
        throughput=_throughput(stats),
        errors=errors or {},
    )


def _status_value(state: BackendState | None, model: dict[str, Any]) -> str:
    """The evaluated status, or a safe read straight off the record.

    The fallback matters: `backend_states` is empty when the DB read failed,
    and a dead backend must still render DEAD rather than inheriting a
    cheerful `status: "ready"` the router left stale.
    """
    if state is not None:
        return state.status.value
    if model.get("backend_alive") is False:
        return BackendStatus.DEAD.value
    return (
        BackendStatus.READY.value
        if model.get("status") == "ready"
        else BackendStatus.NOT_READY.value
    )


def _next_to_evict(models: list[dict[str, Any]], ranks: dict[int, int]) -> int | None:
    """Which unpinned model the router would evict first under pressure.

    Pinned models are exempt from LRU eviction (spec §3), so they are excluded
    outright — including the pinned model with the oldest last_use, which is
    exactly the row a naive "oldest wins" would wrongly condemn.
    """
    candidates = [
        (ranks[i], i) for i, m in enumerate(models)
        if i in ranks and m.get("pinned") is not True
    ]
    if not candidates:
        return None
    return max(candidates)[1]  # highest rank number = least recently used


def _throughput(stats: dict[str, Any] | None) -> ThroughputView | None:
    if not isinstance(stats, dict):
        return None

    rate = _number(stats.get("tokens_per_second"))
    implausible = rate is not None and rate > PLAUSIBLE_MAX_TOKENS_PER_SECOND
    note = None
    if implausible:
        output = stats.get("output_tokens")
        note = (
            f"Lemonade reports {rate:,.0f} tok/s for the last request, which is "
            f"not a real rate — it is what you get dividing a tiny token count "
            f"by a near-zero elapsed time"
            + (f" (the last request produced {output} token(s))." if isinstance(output, int) else ".")
            + " Treat it as no measurement rather than a number."
        )

    return ThroughputView(
        tokens_per_second=rate,
        time_to_first_token=_number(stats.get("time_to_first_token")),
        input_tokens=stats.get("input_tokens"),
        output_tokens=stats.get("output_tokens"),
        request_count_total=stats.get("request_count_total"),
        input_tokens_total=stats.get("input_tokens_total"),
        output_tokens_total=stats.get("output_tokens_total"),
        rate_implausible=implausible,
        rate_note=note,
    )
