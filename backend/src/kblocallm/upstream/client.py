"""The Lemonade proxy — HELM's only outbound path.

FRAME ONLY. No HELM domain logic lives here: there is no `load_model()`, no
`list_models()`, no ctx_size math. Those are the HELM dev's job. What this
gives you is the plumbing that is easy to get wrong and expensive to debug,
pre-solved:

  1. The admin key stays server-side (spec §4, rule 9). Nothing in this
     module returns the key, and the browser never talks to Lemonade.
  2. Per-OPERATION timeout profiles (review F6). One global timeout cannot
     cover /live (milliseconds) and /pull (hours).
  3. Honest error classification (house rule): a 401 from a missing admin
     key, an NPU-exclusivity rejection, a dead backend and a read timeout are
     FOUR different things and must never collapse into "Lemonade error".
  4. Streaming that survives NPM + Cloudflare (review F5).

Spec §4: every route exists under /api/v1/, /api/v0/, /v1/, /v0/. We use
/api/v1/ as canonical. `/live` is root-only — the one exception.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeout profiles — review F6.
#
# httpx has FOUR timeouts (connect/read/write/pool). `timeout=30.0` sets all
# four; a Timeout(...) object with only some fields set silently leaves the
# others at the 5s default. That has cost us real hours in URSA, so every
# profile below is explicit about all four. Pick by INTENT, not by number.
# ---------------------------------------------------------------------------

# Health/liveness polling. If the box doesn't answer in a couple of seconds
# it is not healthy, and the dashboard should say so rather than hang.
FAST = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)

# Control actions. The spec's own §5.3 says a reload drops the model for
# 10-20s, so `read` has to clear that comfortably or HELM will report a
# timeout for a load that actually succeeded — the worst possible lie for a
# control plane (operator retries, second load evicts something).
CONTROL = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)

# Downloads. A 6GB multimodal FLM pull over a home link is minutes-to-hours.
# `read` here is the BETWEEN-BYTES gap, not the total — a healthy download
# never trips it, a stalled one does. That's exactly what we want.
TRANSFER = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)

# Log tail / streaming pulls. Unbounded by design; `read` is the silence
# tolerance. Pair with a keepalive on OUR side (see stream_lines + F5).
STREAM = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

# Non-streaming LLM completions (the Advisor's Summary/Analyze/Chat, the
# benchmark). The 2026.42.1 upgrade made cold prefill at large context ~100s,
# and a 26B reasoning answer adds a minute-plus on top — so `read` has to clear
# ~300s or HELM reports a timeout for an answer that was still coming. The
# connect stays SHORT (15s): a dead or wiped box must fail fast, not hang the
# whole 300s window. Cross-project contract — siblings use the same split
# (docs: lemonade-upgrade-and-provider-contract-for-sibling-devs).
COMPLETION = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=5.0)


class UpstreamErrorKind:
    """Distinct failure classes. Never collapse these into one message.

    This is the direct application of the lesson in
    per-turn-observability-for-sibling-devs.md §4: a 401 surfacing as
    'Network error' cost hours. For HELM the operator needs to tell apart
    'you never configured the admin key' from 'the backend just died'.
    """
    NO_ADMIN_KEY = "no_admin_key"        # HELM has no key; write is impossible
    UNAUTHORIZED = "unauthorized"        # 401/403 — key present but rejected
    NOT_FOUND = "not_found"              # 404 — unknown model/route
    REJECTED = "rejected"                # 4xx — the router said no (bad args, exclusivity)
    UPSTREAM_ERROR = "upstream_error"    # 5xx — Lemonade itself failed
    UNREACHABLE = "unreachable"          # connect failed — lemond down? wrong host?
    TIMEOUT = "timeout"                  # we gave up waiting
    MALFORMED = "malformed"              # 200 but the body isn't what we expect


@dataclass
class UpstreamError(Exception):
    """Carries enough to diagnose without reading the server logs.

    `elapsed_ms` is not decoration: 'failed at 5s' vs 'failed at 180s' vs
    'failed at ~100s' point at three different culprits, and the last one is
    Cloudflare. Instrument the failure, don't guess.
    """
    kind: str
    message: str
    status_code: int | None = None
    elapsed_ms: int | None = None

    def __str__(self) -> str:
        bits = [f"{self.kind}: {self.message}"]
        if self.status_code is not None:
            bits.append(f"HTTP {self.status_code}")
        if self.elapsed_ms is not None:
            bits.append(f"{self.elapsed_ms}ms")
        return " · ".join(bits)


class LemonadeClient:
    """Thin, honest HTTP client for the Lemonade router.

    Construct one per app (wired into app.state.lemonade by main.lifespan) so
    connections are pooled — the dashboard polls every couple of seconds and
    a fresh TCP+TLS handshake per poll is waste.
    """

    def __init__(
        self,
        base_url: str,
        *,
        admin_api_key: str = "",
        api_key: str = "",
    ) -> None:
        self._base = base_url.rstrip("/")
        self._admin_key = admin_api_key
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=self._base)

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def has_admin_key(self) -> bool:
        """Whether write/management calls are possible at all.

        The UI should read this and render 'management disabled — no admin
        key configured' with the write controls HIDDEN, rather than showing
        buttons that mysteriously 401. That's the URSA 1.1.8 lesson: an
        operator debugging a restart loop for an hour because a missing env
        var produced a generic failure.
        """
        return bool(self._admin_key)

    def _headers(self, *, admin: bool) -> dict[str, str]:
        """Build auth headers. The keys NEVER leave this module.

        Spec §4: LEMONADE_API_KEY (currently unset) gates everything;
        LEMONADE_ADMIN_API_KEY (set) gates management routes only.
        """
        headers: dict[str, str] = {"Accept": "application/json"}
        key = self._admin_key if admin else self._api_key
        if not key and not admin:
            key = self._api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # -- core request ------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        admin: bool = False,
        timeout: httpx.Timeout = FAST,
        json_body: dict | None = None,
        params: dict | None = None,
        raw: bool = False,
    ) -> Any:
        """One request, classified failures, no silent fallbacks.

        `admin=True` marks a management/write route (spec §4.2). We refuse
        BEFORE the network call when there's no key — a clear local error
        beats a remote 401 the operator has to interpret.

        `raw=True` returns the response body as text instead of parsing JSON —
        for Lemonade's Prometheus `/metrics`, which is text/plain, not JSON.
        """
        if admin and not self._admin_key:
            raise UpstreamError(
                kind=UpstreamErrorKind.NO_ADMIN_KEY,
                message=(
                    "No Lemonade admin API key configured, so management "
                    "actions are unavailable. Set HELM_LEMONADE_ADMIN_API_KEY "
                    "in the stack's .env (the key lives on the Lemonade host at "
                    "the Lemonade server config) and restart HELM."
                ),
            )

        import time
        t0 = time.monotonic()
        try:
            resp = await self._client.request(
                method,
                path,
                headers=self._headers(admin=admin),
                json=json_body,
                params=params,
                timeout=timeout,
            )
        except httpx.ConnectError as e:
            raise UpstreamError(
                kind=UpstreamErrorKind.UNREACHABLE,
                message=(
                    f"Could not reach Lemonade at {self._base}. Is lemond "
                    f"running on the Lemonade host? ({e})"
                ),
                elapsed_ms=int((time.monotonic() - t0) * 1000),
            ) from e
        except httpx.TimeoutException as e:
            raise UpstreamError(
                kind=UpstreamErrorKind.TIMEOUT,
                # Naming the httpx class matters: ConnectTimeout, ReadTimeout
                # and PoolTimeout mean three different things and the operator
                # (or the next dev) needs to know which one fired.
                message=f"{type(e).__name__} calling {method} {path}",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
            ) from e

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code >= 400:
            body = resp.text[:500]
            if resp.status_code in (401, 403):
                kind = UpstreamErrorKind.UNAUTHORIZED
                msg = (
                    "Lemonade rejected the admin key. Confirm "
                    "HELM_LEMONADE_ADMIN_API_KEY matches the value in "
                    "the Lemonade server config on the Lemonade host."
                    if admin else
                    "Lemonade rejected the request. LEMONADE_API_KEY may now "
                    "be set on the server, which gates ALL routes including "
                    "reads (spec §4)."
                )
            elif resp.status_code == 404:
                kind, msg = UpstreamErrorKind.NOT_FOUND, f"Not found: {path}"
            elif resp.status_code < 500:
                # The router said no for a domain reason — bad args, a
                # reserved arg, an exclusivity conflict. Pass the body
                # through verbatim; it's the most useful thing we have and
                # paraphrasing it would lose the detail.
                kind, msg = UpstreamErrorKind.REJECTED, f"Lemonade rejected it: {body}"
            else:
                kind, msg = UpstreamErrorKind.UPSTREAM_ERROR, f"Lemonade error: {body}"
            raise UpstreamError(
                kind=kind, message=msg,
                status_code=resp.status_code, elapsed_ms=elapsed_ms,
            )

        if not resp.content:
            return None
        if raw:
            return resp.text
        try:
            return resp.json()
        except ValueError as e:
            raise UpstreamError(
                kind=UpstreamErrorKind.MALFORMED,
                message=f"Expected JSON from {path}, got: {resp.text[:200]}",
                status_code=resp.status_code, elapsed_ms=elapsed_ms,
            ) from e

    # -- streaming ---------------------------------------------------------

    async def stream_lines(
        self,
        method: str,
        path: str,
        *,
        admin: bool = False,
        json_body: dict | None = None,
    ) -> AsyncIterator[str]:
        """Yield upstream lines as they arrive (log tail, streaming pull).

        This is only HALF of surviving the ingress. The other half is on the
        way OUT to the browser, and it is NOT optional (review F5):

          - Wrap this in a StreamingResponse with `X-Accel-Buffering: no`
            and `Cache-Control: no-cache`.
          - Emit a keepalive line every ~15s when this iterator is QUIET.
            Cloudflare kills a response after ~100s with no bytes; a log tail
            at 3am is silent for far longer than that. The keepalive is what
            defeats 524 — it is one line and it is the whole fix.
          - NPM needs `proxy_buffering off` + `proxy_http_version 1.1` in the
            proxy host's Custom Nginx Configuration box, or nginx holds the
            stream and none of the above is visible at the edge.

        See infra/npm-mtls-ingress.md for the paste-ready config, and
        C:\\dev\\fitbit-web-ui-app\\docs\\sse-streaming-for-sibling-devs.md for
        the full explanation.
        """
        if admin and not self._admin_key:
            raise UpstreamError(
                kind=UpstreamErrorKind.NO_ADMIN_KEY,
                message="No Lemonade admin API key configured.",
            )
        async with self._client.stream(
            method,
            path,
            headers=self._headers(admin=admin),
            json=json_body,
            timeout=STREAM,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise UpstreamError(
                    kind=UpstreamErrorKind.REJECTED,
                    message=f"Stream refused: {body}",
                    status_code=resp.status_code,
                )
            async for line in resp.aiter_lines():
                yield line

    # -- the two calls every HELM view needs -------------------------------
    #
    # Intentionally the ONLY named endpoints in the frame. They're here to
    # show the shape (path, admin flag, timeout profile) and to power the
    # connectivity check on the system page. Everything else in spec §4 is
    # the HELM dev's to add.

    async def live(self) -> bool:
        """GET /live — root-only (spec §4). Does no work; safe to poll hard."""
        try:
            await self.request("GET", "/live", timeout=FAST)
            return True
        except UpstreamError:
            return False

    async def health(self) -> dict:
        """GET /api/v1/health — the core read (spec §4.1).

        Returns the raw router payload. Deliberately NOT parsed here: the
        ctx_size-per-slot derivation (spec rule 1) belongs in a tested pure
        function, not buried in a client method. See review F4 — that
        derivation is the highest-risk code in the product and it needs a
        table-driven test, which means it needs to be its own module.
        """
        return await self.request("GET", "/api/v1/health", timeout=FAST)
