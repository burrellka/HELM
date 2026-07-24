"""Phase 3: the model library + downloads. Spec §5.2, §5.3a, §5.5.

Reads are cheap and cacheable; the catalog is 162 entries on this box and
changes only when something is pulled or deleted.

**Downloads are server-owned on purpose.** `POST /api/v1/pull` with
`stream: true, subscribe: false` hands the job to Lemonade and returns; HELM
then polls `GET /v1/pull/jobs`. That is what makes a 6GB pull survive a
browser refresh — the entire point of §5.5. HELM does NOT proxy the download
stream through itself: doing so would tie the job's life to one HTTP
connection through NPM and Cloudflare, which is precisely the thing that 524s.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..advisor_service import HFMetadataService, ModelSourceStore
from ..domain.drift import DriftReport
from ..domain.hf_resolve import RepoConfidence, resolve_repo
from ..domain.library import LabelJoin, build_library, search, unmatched_loaded_models
from ..storage.metadata import MetadataStore
from ..upstream.client import CONTROL, FAST, TRANSFER, LemonadeClient, UpstreamError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/library", tags=["library"])

#: The catalog is 162 entries of mostly-static metadata. It changes on pull
#: and delete, both of which invalidate it explicitly, so a short TTL is just
#: insurance against a change made outside HELM (the CLI).
CATALOG_TTL_SECONDS = 60.0


class DriftFieldOut(BaseModel):
    field: str
    kind: str
    saved: Any = None
    running: Any = None
    benign: bool = False
    note: str | None = None


class DriftOut(BaseModel):
    has_drift: bool
    has_unpersisted: bool
    fields: list[DriftFieldOut] = Field(default_factory=list)


class SettingsOut(BaseModel):
    """One side of §5.3's editor, in the operator's units.

    `ctx_per_slot` is derived server-side by the same code that powers the
    dashboard's Concurrency column. The frontend prefills the form from this
    and never divides anything itself (F4: one implementation, one test table).
    """

    slots: int | None = None
    ctx_per_slot: int | None = None
    ctx_size: int | None = None
    backend: str | None = None
    #: Sampling params lifted out of llamacpp_args into their own fields, so the
    #: editor shows them as labeled inputs and `extra_args` is what remains.
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    extra_args: str = ""


class EntryOut(BaseModel):
    model_id: str
    recipe: str | None = None
    checkpoint: str | None = None
    downloaded: bool = False
    suggested: bool = False
    update_available: bool = False
    size: int | None = None
    max_context_window: int | None = None

    labels: list[str] = Field(default_factory=list)
    raw_labels: list[str] = Field(default_factory=list)
    #: matched / no_labels / unknown. The UI renders "capabilities unknown"
    #: for `unknown` and NEVER an empty badge row (F9).
    label_join: str = LabelJoin.UNKNOWN.value
    #: True when wiring an agent to this model fails SILENTLY (Rule 10).
    strips_tools_silently: bool = False

    loaded: bool = False
    pinned: bool = False
    device: str | None = None

    description: str = ""
    favorite: bool = False
    archived: bool = False
    tags: list[str] = Field(default_factory=list)

    drift: DriftOut | None = None
    saved_settings: SettingsOut | None = None
    running_settings: SettingsOut | None = None


class LibraryOut(BaseModel):
    entries: list[EntryOut] = Field(default_factory=list)
    known_tags: list[str] = Field(default_factory=list)
    total: int = 0
    #: Loaded models with no catalog row — the F9 join failing in the wild.
    #: Surfaced so the operator knows which rows HELM cannot vouch for.
    unmatched_loaded: list[str] = Field(default_factory=list)


def _drift_out(report: DriftReport | None) -> DriftOut | None:
    if report is None:
        return None
    return DriftOut(
        has_drift=report.has_drift,
        has_unpersisted=report.has_unpersisted,
        fields=[
            DriftFieldOut(
                field=f.field, kind=f.kind.value, saved=f.saved,
                running=f.running, benign=f.benign, note=f.note,
            )
            for f in report.fields
        ],
    )


class _CatalogCache:
    def __init__(self) -> None:
        self._value: dict | None = None
        self._at: float = 0.0

    def get(self) -> dict | None:
        if self._value is not None and (time.monotonic() - self._at) < CATALOG_TTL_SECONDS:
            return self._value
        return None

    def put(self, value: dict) -> None:
        self._value, self._at = value, time.monotonic()

    def invalidate(self) -> None:
        self._at = 0.0


def _cache(request: Request) -> _CatalogCache:
    cache = getattr(request.app.state, "catalog_cache", None)
    if cache is None:
        cache = _CatalogCache()
        request.app.state.catalog_cache = cache
    return cache


async def _catalog(request: Request, force: bool = False) -> dict:
    client: LemonadeClient = request.app.state.lemonade
    cache = _cache(request)
    if not force:
        cached = cache.get()
        if cached is not None:
            return cached
    try:
        # NOTE: include_paths is NOT passed (F11). It publishes the host's
        # filesystem layout, and a control plane should not leak that as a
        # side effect of listing models. It is an explicit per-request
        # operator action on the files endpoint only.
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST,
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=f"{e.kind}: {e}")
    cache.put(payload)
    return payload


@router.get("", response_model=LibraryOut)
async def library(
    request: Request,
    text: str = "",
    labels: str = "",
    tags: str = "",
    recipes: str = "",
    downloaded: bool | None = None,
    favorite: bool | None = None,
    loaded: bool | None = None,
    include_archived: bool = False,
) -> LibraryOut:
    """The catalog, joined with loaded state and the operator's own metadata.

    Filtering happens server-side because `domain.library.search` is where the
    §5.3a query semantics are tested. The UI sends intent, not predicates.
    """
    client: LemonadeClient = request.app.state.lemonade
    catalog = await _catalog(request)
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
    except UpstreamError:
        # Losing health costs the loaded/pinned columns and the drift report.
        # The catalog is still worth showing.
        health = None

    store = MetadataStore(request.app.state.db)
    metadata = store.all()
    entries = build_library(catalog, health, metadata)

    def _split(value: str) -> set[str]:
        return {v.strip() for v in value.split(",") if v.strip()}

    results = search(
        entries,
        text=text,
        labels=_split(labels),
        tags=_split(tags),
        recipes=_split(recipes),
        downloaded=downloaded,
        favorite=favorite,
        loaded=loaded,
        include_archived=include_archived,
    )
    return LibraryOut(
        entries=[
            EntryOut(
                **{
                    k: v for k, v in e.__dict__.items()
                    if k not in ("drift", "label_join", "saved_settings", "running_settings")
                },
                label_join=e.label_join.value,
                strips_tools_silently=e.strips_tools_silently,
                drift=_drift_out(e.drift),
                saved_settings=(
                    SettingsOut(**e.saved_settings.__dict__) if e.saved_settings else None
                ),
                running_settings=(
                    SettingsOut(**e.running_settings.__dict__) if e.running_settings else None
                ),
            )
            for e in results
        ],
        known_tags=store.known_tags(),
        total=len(entries),
        unmatched_loaded=unmatched_loaded_models(catalog, health),
    )


class MetadataBody(BaseModel):
    """All fields optional. Omitted means "leave alone", not "clear"."""

    description: str | None = None
    favorite: bool | None = None
    archived: bool | None = None
    tags: list[str] | None = None


@router.patch("/{model_id:path}/metadata")
def update_metadata(model_id: str, body: MetadataBody, request: Request) -> dict:
    """§5.3a. HELM-local; needs no admin key and touches Lemonade not at all.

    Note this is NOT gated on `has_admin_key`: it writes to HELM's own
    database, not the fleet. A read-only HELM (no admin key) can still be the
    operator's notebook, and refusing that would be gatekeeping for no reason.
    """
    store = MetadataStore(request.app.state.db)
    return store.update(
        model_id,
        description=body.description,
        favorite=body.favorite,
        archived=body.archived,
        tags=body.tags,
    )


@router.get("/export")
def export_metadata(request: Request) -> dict:
    """§5.3a: portable, backup-able. The only copy of the operator's own
    thinking about 161 models — it must be possible to get it out."""
    return MetadataStore(request.app.state.db).export()


@router.post("/import")
def import_metadata(payload: dict, request: Request) -> dict:
    """Additive merge. Deliberately does not delete anything absent from the
    payload — an import is not a restore-to-snapshot, and treating it as one
    would silently destroy tags added since the export."""
    count = MetadataStore(request.app.state.db).import_(payload)
    return {"imported": count}


# ── downloads (§5.5) ────────────────────────────────────────────────────────


def _require_admin(request: Request) -> LemonadeClient:
    client: LemonadeClient = request.app.state.lemonade
    if not client.has_admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HELM has no Lemonade admin key, so pull and delete are unavailable. "
                "Set HELM_LEMONADE_ADMIN_API_KEY in the stack's .env and restart HELM."
            ),
        )
    return client


class PullBody(BaseModel):
    model_name: str
    checkpoint: str | None = None
    recipe: str | None = None


@router.post("/pull")
async def pull(body: PullBody, request: Request) -> dict:
    """Start a download. Server-owned, so it survives a browser refresh.

    `stream: true, subscribe: false` is the load-bearing combination (§4.2):
    the job becomes Lemonade's, queryable via /v1/pull/jobs, and it keeps
    running when this request returns and when the operator closes the tab.

    Custom models must use the `user.` namespace (§4.2) to avoid colliding
    with built-ins. We warn rather than rewrite the name: silently renaming
    what the operator typed is its own surprise.
    """
    client = _require_admin(request)
    warnings: list[str] = []

    # Lemonade's rule (verified against the box): `checkpoint`/`recipe` are ONLY
    # accepted when the model name is `user.`-prefixed — a pull that sends them
    # for a built-in catalog model is a hard 400. So for a built-in (the common
    # case: the Library's Download button on a catalog row), pull BY NAME and let
    # Lemonade resolve the checkpoint from its own catalog. Sending the
    # catalog-provided checkpoint back to Lemonade is exactly what broke the
    # Download button (the 400 surfaced as a 502).
    is_custom = body.model_name.startswith("user.")
    payload: dict[str, Any] = {
        "model_name": body.model_name, "stream": True, "subscribe": False,
    }
    if is_custom:
        if body.checkpoint:
            payload["checkpoint"] = body.checkpoint
        if body.recipe:
            payload["recipe"] = body.recipe
    elif body.checkpoint or body.recipe:
        # A built-in was given a checkpoint/recipe (the frontend passes the
        # catalog checkpoint for convenience). Drop them — Lemonade owns the
        # built-in's checkpoint — and say so rather than 400.
        warnings.append(
            "Pulling the built-in by name; ignored the supplied checkpoint/recipe "
            "(Lemonade resolves those from its own catalog for built-ins). To pull "
            "a custom model with your own checkpoint, prefix the name with 'user.'."
        )

    try:
        result = await client.request(
            "POST", "/api/v1/pull", admin=True, json_body=payload, timeout=TRANSFER,
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail={"kind": e.kind, "detail": str(e)})

    # Remember where this model came from. Lemonade will report a local file path
    # for it after the download, losing the HF origin; recording the pull-time
    # checkpoint lets the advisor still resolve the real repo (migration 11).
    if body.checkpoint:
        from ..advisor_service import ModelSourceStore
        ModelSourceStore(request.app.state.db).set(
            body.model_name, body.checkpoint, body.recipe)

    _cache(request).invalidate()
    return {"started": True, "result": result, "warnings": warnings}


@router.get("/jobs")
async def jobs(request: Request) -> dict:
    """Download jobs — what makes a server-owned pull survive a browser reload,
    and what the header download indicator polls.

    The live endpoint is `GET /api/v1/downloads` (the spec's `/v1/pull/jobs`
    404s on the shipped Lemonade — verified against the box). Each job carries
    `model_name`, `percent`, `running`, `status`, `error`, and byte counters
    (`overall_bytes_downloaded` / `total_download_size`). We return the raw jobs
    plus a normalized `active` view (running, not complete, no error) so the UI
    does not have to re-derive "what's downloading right now" from a mixed list
    that also holds finished and failed jobs. Download SPEED is not in the
    payload; the client derives it from byte deltas between polls.
    """
    client: LemonadeClient = request.app.state.lemonade
    try:
        raw = await client.request("GET", "/api/v1/downloads", timeout=FAST)
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail={"kind": e.kind, "detail": str(e)})

    jobs_list = raw if isinstance(raw, list) else (raw.get("downloads") or raw.get("jobs") or [])
    active = [
        {
            "id": j.get("id"),
            "model_name": j.get("model_name") or j.get("id"),
            "percent": j.get("percent"),
            "bytes_downloaded": j.get("overall_bytes_downloaded") or j.get("bytes_downloaded"),
            "bytes_total": j.get("total_download_size") or j.get("bytes_total"),
            "status": j.get("status"),
        }
        for j in jobs_list
        if isinstance(j, dict) and j.get("running") and not j.get("complete") and not j.get("error")
    ]
    return {"jobs": jobs_list, "active": active, "active_count": len(active)}


@router.get("/card/{model_id:path}")
async def model_card(model_id: str, request: Request) -> dict:
    """The HuggingFace model card (README) for the download/update preview, so the
    operator can see what a model IS before pulling gigabytes of it.

    HF's JSON metadata carries no usable description; the card README does. We
    resolve the repo the same way Analyze does — from the catalog checkpoint, or
    the checkpoint HELM remembered at pull time when Lemonade now reports a local
    path — then fetch and cache the card. Honest about absence: a custom/local
    model with no HF repo returns no card, not an invented one.
    """
    client: LemonadeClient = request.app.state.lemonade
    # Find the model's checkpoint from the catalog.
    checkpoint = None
    try:
        payload = await client.request(
            "GET", "/api/v1/models", params={"show_all": "true"}, timeout=FAST)
        for m in payload.get("data", []):
            if isinstance(m, dict) and m.get("id") == model_id:
                checkpoint = m.get("checkpoint")
                break
    except UpstreamError:
        pass

    resolution = resolve_repo(checkpoint, model_id)
    if not (resolution.confidence is RepoConfidence.EXPLICIT and resolution.repo_id):
        remembered = ModelSourceStore(request.app.state.db).get(model_id)
        if remembered:
            resolution = resolve_repo(remembered, model_id)
    if not (resolution.confidence is RepoConfidence.EXPLICIT and resolution.repo_id):
        return {"repo_id": None, "card": None,
                "note": "No HuggingFace repo for this model, so there is no card to show "
                        "(a custom or local-file model)."}

    card = await HFMetadataService(request.app.state.db).fetch_card(resolution.repo_id)
    return {
        "repo_id": resolution.repo_id,
        "card": card,
        "note": None if card else "HuggingFace has no README card for this repo.",
    }


class DeleteBody(BaseModel):
    model_name: str
    #: Deleting weights is irreversible and can be 20GB of re-download over a
    #: home link. An explicit confirm is the cheapest possible guard.
    confirm: bool = False


@router.post("/delete")
async def delete(body: DeleteBody, request: Request) -> dict:
    """Delete a downloaded model. Irreversible; hence the confirm."""
    client = _require_admin(request)
    if not body.confirm:
        raise HTTPException(status_code=409, detail={
            "code": "needs_confirmation",
            "message": (
                f"Deleting {body.model_name} removes its weights from disk. This cannot be "
                f"undone and re-downloading may be many GB. Send confirm=true to proceed."
            ),
        })
    try:
        health = await client.request("GET", "/api/v1/health", timeout=FAST)
        loaded = {
            m.get("model_name") for m in (health.get("all_models_loaded") or [])
            if isinstance(m, dict)
        }
        if body.model_name in loaded:
            # Refuse rather than find out what the router does with weights
            # that vanish under a running backend.
            raise HTTPException(status_code=409, detail={
                "code": "loaded",
                "message": (
                    f"{body.model_name} is currently LOADED. Unload it before deleting its "
                    f"weights."
                ),
            })
    except UpstreamError:
        logger.warning("Could not verify loaded state before delete of %s", body.model_name)

    try:
        result = await client.request(
            "POST", "/api/v1/delete", admin=True,
            json_body={"model_name": body.model_name}, timeout=CONTROL,
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail={"kind": e.kind, "detail": str(e)})

    _cache(request).invalidate()
    return {"deleted": True, "result": result}
