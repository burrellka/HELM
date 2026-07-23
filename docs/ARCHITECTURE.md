# HELM — Architecture

HELM is a self-hosted web control plane over [Lemonade Server](https://github.com/lemonade-sdk/lemonade), a local-LLM runtime. It gives an operator one honest screen for a fleet of local models: what is loaded, how much memory the GPU pool actually holds, how many concurrency slots a model really has, whether a backend has crashed, and a grounded Model Advisor for reasoning about which model to run.

The guiding principle throughout is **an honest number or a visible `unknown`, never a plausible-looking guess**. A control plane that reports a confident wrong value is worse than no UI at all — the operator acts on it. Most of the design decisions below trace back to a specific failure that a confident guess caused, and the code says so at each site.

This document describes the real implementation. It contains no deployment specifics on purpose (this is a public repo): substitute *your Lemonade host*, *your data directory*, and *your operator password* for the placeholders.

---

## 1. Overview and topology

HELM is two containers plus one bind-mounted volume, sitting in front of a Lemonade Server you already run.

- **`helm-web`** — an nginx image serving the built React/Vite/TypeScript single-page app and reverse-proxying `/api` to the backend. It owns nothing on disk.
- **`helm-api`** — a `uvicorn` process running the FastAPI (Python 3.12) backend. It is the only component that talks to Lemonade, and the only component that holds secrets.
- **One `/data` volume** — a host directory bind-mounted into `helm-api`, holding HELM's own SQLite database, the JWT signing secret, and the operator password hash. Nothing about the fleet lives here; only HELM's own state (see §7).

### Request flow

```
browser ──▶ helm-web (nginx) ──▶ helm-api (uvicorn/FastAPI) ──▶ Lemonade Server
             serves SPA            domain logic + SQLite         (your Lemonade host)
             proxies /api          holds the admin key
```

Every browser request is same-origin: one nginx serves the SPA *and* proxies the API, so there is no cross-origin caller. The backend deliberately installs **no CORS middleware** — a permissive CORS policy on a write-capable control plane would hand any site the operator visits the ability to drive the fleet. If HELM ever appears to need CORS, the origins have been split by accident.

### Two nginx blocks that exist because of real incidents

The `helm-web` config carries two pieces that look trimmable and are not:

**The resolver trick.** nginx resolves an upstream name *once* at config load and caches the IP forever. Restart `helm-api`, it gets a new IP on the container network, and nginx keeps proxying to the dead one — 502s that "fix themselves" when you restart the web container, sending everyone hunting in the wrong place. The fix is to force runtime DNS resolution by proxying through a variable:

```nginx
resolver 127.0.0.11 valid=30s ipv6=off;   # the container runtime's embedded DNS
map "" $helm_api_upstream { default "helm-api"; }
...
location /api/ {
    proxy_pass http://$helm_api_upstream:8000;   # variable ⇒ resolved per-request
}
```

**Streaming with buffering off.** nginx buffers proxied responses by default, so a streamed response never flushes and the UI shows nothing until the connection dies — then an edge proxy 524s it after ~100s of silence. Both are invisible in local dev and only appear behind real ingress. The `/api/` location sets `proxy_buffering off`, `proxy_cache off`, `proxy_http_version 1.1` (chunked transfer needs 1.1), and `proxy_read_timeout 600s` (a model load takes 10–20s and a pull takes minutes; the 60s default would kill both). It also forwards `X-Forwarded-Proto`, which the backend keys the session cookie's `Secure` flag off of (see §8).

The SPA's `index.html` is served `no-store` while the content-hashed JS/CSS bundles are cached `immutable` — cache `index.html` and a browser can run an entire stale app from disk after a deploy.

---

## 2. The shared `llamacpp_args` parser — the highest-risk component

`domain/llamacpp_args.py` parses the free-text `llamacpp_args` string an operator can edit on a model's recipe options. It is the single most dangerous function in the product, and it is worth being blunt about why.

### The 8× lie

HELM divides a model's total `ctx_size` by whatever `--parallel` value it finds in the args to compute the *per-slot* context. A naive parser that only looks for `--parallel` misses the short form `-np` — which is the form most people actually type. On a miss, the old code fell back to "default 1", so `per_slot = ctx_size = 524288` and the dashboard displayed **eight times the real context, with total confidence**, for a model running `-np 8`.

The contract that prevents this: **a value we are sure of, or a visible `unknown`. Never a plausible number.**

### The FOUND / ABSENT / MALFORMED tri-state

`lookup_int` returns an `IntFlag` carrying one of three statuses, and the distinction between the last two is load-bearing:

- **`FOUND`** — the flag was present and parsed to a usable integer.
- **`ABSENT`** — the flag was not there at all. This is a *real answer*: llama.cpp's documented default for `--parallel` is 1, so an args string without it genuinely means one slot. The caller renders it as `1 slot (default)` — quoting the default back, not reporting a setting the operator made.
- **`MALFORMED`** — the operator typed something unreadable (`--parallel abc`, a dangling `--parallel` with no value, `--parallel 0` which would divide by zero, an unbalanced quote). This must reach the screen as `unknown`. It must **never** collapse into `ABSENT`, because `ABSENT` produces a confident number and `MALFORMED` must produce a confessed non-answer.

The tokenizer reinforces this: an untokenizable string (unbalanced quotes) returns `None`, not `[]`. An empty list looks exactly like "operator set no flags" and would hand back the 8× bug through the back door; `None` forces every lookup to degrade to `MALFORMED`.

### Last-wins on duplicates

`--parallel 8 --parallel 4` really does run 4 slots — llama.cpp parses argv left to right and the later assignment overwrites the earlier. So `lookup_int` takes the *last* occurrence. Picking the first would be a confident lie about a string the operator can read for themselves.

A subtlety worth noting: the next token after a flag is treated as its value only if it does not itself look like a flag — but `-1` and `-0.5` are *negative values*, not flags. `_looks_like_flag` runs a float check precisely so `--parallel -1` is read as a (malformed, non-positive) value rather than a bare flag followed by a mystery.

### One parser, two consumers

This is the architectural point. The same module serves:

1. **The concurrency derivation** (`domain/concurrency.py`, §3) — `parallel_slots` for the per-slot context math.
2. **The reserved-arg validator** (`find_reserved`) — which must reject `--ctx-size`, `-c`, *and* `--ctx-size=4096`, the same three spellings this module already normalizes.

Both go through the one `find_flag` primitive that knows those three spellings are the same flag. The alternative — a second hand-rolled arg parser (say, a regex in a React component that "removes `--parallel`") — is a known bug class: two hand-maintained parsers drift, and the one that drifts gets `-np` subtly wrong the first time someone types the short form, silently. `strip_flag` and the sampling-flag extraction (temperature, top-k, top-p, min-p, repeat-penalty) all route through the same finder for exactly this reason. **If you need a new flag read, add an alias tuple and a test row here — do not write a second parser.**

---

## 3. Reading the fleet truthfully

Four domain modules turn Lemonade's raw payloads into claims HELM is willing to stand behind. Each is pure and table-tested, so the string the operator reads is the string a test asserts.

### Memory pool — the real GPU pool, not a carve-out (`domain/pool.py`)

A wrong number here once cost months. An app-layer bug read a small BIOS carve-out instead of the large unified GPU memory pool, reported a phantom ceiling, and everything downstream reasoned from it. The trap is still in the payload and is nastier than it sounds: the GPU device entry carries *both* numbers side by side, and the wrong one has the friendlier name (`vram_gb`). There are two further decoys — a top-level physical-RAM string that is close enough to look right, and a same-named `vram_gb` field on a *different* endpoint that means current usage, not capacity.

`extract_pool` is the one place the denominator is decided. It reads the pool per poll (no constant is baked in — the illustrative figure lives only in a test fixture), refuses to fall back to the carve-out field, and returns `capacity_gb = None` (rendered `unknown`, never `0`) whenever it cannot read a usable number. It also cross-checks the `enable_dgpu_gtt` flag: the pairing "pool reads small **and** GTT is disabled" is the original saga's exact signature, so HELM catches in one glance what took months, and emits a specific, actionable warning (`SMALL_POOL_GTT_DISABLED`) with the fix. Multiple available GPUs return `AMBIGUOUS_DEVICE` rather than drawing a confident bar against one arbitrary denominator. The `MemoryPool.percent` property returns `None` whenever either half is unknown — a bar drawn against a guessed denominator is the exact thing this module exists to prevent. The field path it actually read is carried in `source` and shown in the UI tooltip, so "where does this number come from?" is answered on screen, not by code archaeology.

### Concurrency and per-slot context (`domain/concurrency.py`)

Two rules meet here, both because the CLI dashboard HELM replaces got them wrong.

**Slots are an allowlist, not a denylist.** The old shape was `if flm: SERIAL else: slots`, which fabricates a slot count for every recipe that is neither llamacpp nor flm. A live box carries ten-plus recipes (whispercpp, sd-cpp, kokoro, vllm, …), so that `else` branch was a loaded gun. In HELM, `llamacpp` is the *only* recipe that derives slots — `SLOTTED_RECIPES = {"llamacpp"}`. `flm` (the NPU runtime) is explicitly `SERIAL` by router construction. Everything else is honestly `n/a` — not a fabricated count, and not a `SERIAL` claim never verified for that backend.

**`ctx_size` is a TOTAL, divided across slots.** A model reading `ctx_size: 524288` with `--parallel 8` is 8 slots of 65,536, not 8 slots of 524,288. This module only ever emits the *derived* per-slot value, and emits `unknown` rather than a computed one whenever the `parallel_slots` parse it depends on returned `MALFORMED`. That single line — `MALFORMED` becomes `UNKNOWN`, never `default 1` — is the whole reason §2 exists.

### Crash detection via pid persistence (`domain/crash.py`, migrations 1–2)

The obvious signal — Lemonade's `watchdog_reset` flag — is a trap. An induced crash on a live box (a `SIGKILL` of a backend) left `watchdog_reset` *false* throughout, because the next request lazily reloaded the backend before the watchdog ever polled it. On a busy model that is the normal case, which makes the flag reliable only for crashes nobody noticed.

So the detector is the **pid**. Each loaded backend is its own OS process; if the pid under a model changes between polls, that process died and a new one replaced it. HELM persists the last-seen pid in SQLite (`backend_state`, migration 2) and compares. `watchdog_reset` corroborates when present but never leads. Two guards keep it honest:

- A restart is only claimed when *both* the previous and current pids are known — the first sighting of a model is never a restart, so a container start does not cry wolf. An unknown pid is stored as `NULL`, never `0`, so two unknowns do not compare unequal and manufacture a phantom restart.
- The wording is careful. A pid change proves a *restart*, not a *crash* — someone running `lemonade load` at the CLI produces identical evidence. The UI says "restarted" (what HELM can actually see), not "CRASHED". Claiming a crash on the operator's own deliberate reload is exactly the confident wrongness the product exists to avoid. A restart older than a 15-minute window is shown with its timestamp but not shouted about — a permanent warning teaches the operator to ignore warnings, which costs the next incident.

### The capability-label join — "no tools" vs "unknown" (`domain/library.py`)

On the NPU, tool calling is compiled per-model into the kernel, and the FLM runtime **silently strips the `tools` array** for models that lack it — the model never sees the tools and hallucinates an answer instead. The `tool-calling` label is the only reliable indicator; not the name, not the size, not the family. Wiring an agent to a model that silently strips tools is a real, quiet failure.

The join key is `health.all_models_loaded[].model_name == models.data[].id` (live-confirmed). But a model showing no `tool-calling` badge because the *join failed* looks identical to one that genuinely lacks the capability — absence of evidence is not evidence of absence. So the join is **tri-state**:

- **`MATCHED`** — joined, and the model publishes capability labels.
- **`NO_LABELS`** — joined, and it genuinely publishes none. A real, load-bearing answer: an FLM model here *will* strip tools.
- **`UNKNOWN`** — the join failed. We know nothing, and the row renders "capabilities unknown", never an empty badge strip.

The distinction is enforced at the source: `labels: []` from the server (`NO_LABELS`) is not the same as no `labels` key at all (`UNKNOWN`). The `strips_tools_silently` property only fires on a successful join over a tool-stripping recipe — asserting "this strips tools" on a *failed* join would be a confident lie in the opposite direction. Non-chat models (embeddings, reranking) are excluded from the warning entirely: a warning that fires where it cannot matter is how the warning that does matter gets ignored.

---

## 4. The Model Advisor

The Advisor (`advisor_service.py`, `api/advisor.py`) reasons about a single model — what it is, whether it fits this box, what to run instead — for an operator who is new to local models. Its entire trustworthiness rests on **three provenance layers, each labeled distinctly** so the UI never presents generated text as a fact.

- **Layer 1 — deterministic local facts** (`domain/advisor.py`, pure). Parameter count, quantization class and byte-per-weight, size, fit verdict against the detected pool. No model runs; the facts card renders instantly and is always present.
- **Layer 2 — cached external metadata** (`HFMetadataService`). A server-side HuggingFace lookup (the browser never reaches the open web — a LAN-only control plane does not get arbitrary egress), cached in SQLite by resolved repo id with a one-week TTL, rate-limited. Crucially it degrades to a truthful "no external record" rather than inventing a card, and a network failure is reported as "could not check", distinct from a genuine 404 "no record" (which is itself cached so it is not re-fetched every time).
- **Layer 3 — LLM synthesis, labeled generated.** The analysis model runs over Layers 1 and 2 *only*, and its output is always tagged "Generated by *model* over the facts above. Not an independent source."

### The locked safety kernel vs editable instructions

Layer 3's system prompt is two layers. The **operational prompt** is a code constant — un-editable, prepended before anything else — and it is deliberately minimal: never state a model-specific fact (benchmark, parameter count, context length, license, date) unless it appears in the provided context; never fabricate a source; ground all hardware reasoning in the *detected* hardware, not example specs. It exists so that even a botched instruction edit — or a bad paste — cannot switch off the anti-fabrication floor for a novice who cannot spot a confident lie.

Everything rich — teaching approach, sources of truth, search discipline — lives in the **editable instructions** layer, which ships with the architect's canonical text as a default and is overridable from the UI (stored in the DB; the override wins). The knowledge primer (a vetted "how local models work" reference) is attached the same way and injected as an authoritative section. The settings endpoint returns `operational_prompt_locked: true` so the UI can show that the kernel cannot be edited.

The prompt assembly (`build_synthesis_prompt`) grounds every source explicitly: detected hardware first (so the advice is portable across machines — see §6), the target model's local facts, external metadata with honest absence, the operator's *actual owned inventory* as the candidate set for "recommend something else" (so it cannot invent a model that is not owned), and web-search results labeled either real-and-cited or `not_run` — never "nothing was found" when a search simply did not happen.

### Summary vs Analyze — a deliberate split

Two paths, not one call with a flag:

- **Summary** (§1.10) is a fast, **starved-context** read of one model. It builds its own tiny prompt — the facts card *only*, no inventory, no web search, no knowledge primer, not even hardware — behind its own locked floor, runs its own small model (a sub-1B by default), and caps output low. The starvation is what keeps a tiny model grounded (it cannot reach for anything off the card) and short. The live measurement settled the design: the small model returns in ~1.6s where the heavy synthesis took ~2 minutes.
- **Analyze / deep synthesis** is the heavy teach: the big analysis model, the full grounded context, owned-inventory comparison, optional web search. It is slow because it does real work.

A small model is the *right* tool for Summary and the *wrong* tool for Analyze, and that is not a contradiction: Analyze needs judgment (small models fabricate under that load), Summary only restates the card. The picker reflects this — sub-2B models are flagged "fabricates specs in this role" for the analyst but recommended small-first for Summary.

Everything the Advisor emits is **grounded-never-recalled**: on any failure (model not loaded, inference error, empty output) it returns a labeled "unavailable" and leaves the facts card standing, rather than papering over the gap with a fabricated write-up.

---

## 5. Streaming the Advisor chat

`POST /api/v1/advisor/chat/stream` relays the analysis model's tokens as Server-Sent Events so the operator sees words in a second or two instead of staring at a spinner for the ~1–2 minutes a reasoning answer takes. Several details are load-bearing.

**Three event kinds.** The analysis model is a reasoning model: it emits a hidden reasoning channel *first*, then the answer. HELM relays reasoning as its own `reasoning` event (rendered as a live "thinking" indicator, never as the answer — surfacing the scratchpad as the reply would be its own kind of lie), then `text` deltas for the real reply, then a final `complete` event carrying usage, wall-clock elapsed, and the model id for the per-turn line. If the model produced reasoning but no content and finished on `length`, the `complete` event carries an honest note that the token budget was eaten by reasoning and points at a non-reasoning analyst.

**A 15s keepalive — and why the naive version fails.** A byte must cross the wire every ~15s or an intermediary reads the silence of a long generation as a dead origin and kills it — and the cold first token can land right on top of that window. The obvious implementation is to wrap the source generator's `__anext__` in `asyncio.wait_for(..., timeout=15)`. **This tears down the stream.** When the timeout fires, `wait_for` *cancels* that coroutine, and the cancellation propagates into the generator's in-flight httpx read, killing the upstream connection — so after the first keepalive, no token ever arrives.

The fix is a **producer task draining the source into a queue**; only the queue `get` is cancelled on timeout, never the source itself:

```python
async def producer():
    async for event in source:
        await queue.put(("data", event))
    await queue.put((_DONE, None))

task = asyncio.create_task(producer())
while True:
    try:
        kind, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
    except asyncio.TimeoutError:
        yield ": keepalive\n\n"      # the upstream read is untouched
        continue
    ...
```

The response also sets `X-Accel-Buffering: no` and `Cache-Control: no-cache` — belt to the nginx suspenders from §1 — and delivers upstream failures *inside* the stream as an `error` event, because the connection is already open and raising would just drop it as "stream ended".

**Stable-prefix ordering.** The grounded prompt is assembled in a stable order (kernel, instructions, primer, then context). A stable prefix lets llama.cpp's prompt cache hit across turns, so only the changing tail is re-processed — the cold prefill is paid once, not every message.

---

## 6. Portability — one image, N boxes

`domain/system_profile.py` reads all hardware **live** from Lemonade's `system-info` and `system-stats`. Nothing about any specific machine is baked into code or prose: processor, core/thread counts, GPU name and family, the memory pool (via the one `extract_pool` source of truth from §3), NPU family and TOPS, RAM, OS version, storage, and which inference backends are installed vs installable. Deploy the same image to a different box and the panel reports *that* box. When `system-info` cannot be read, the profile is marked `detected: false` and the UI says so rather than showing a fabricated machine.

This is what makes the Advisor portable too: `advisor_hardware_context` hands the model the *detected* numbers and instructs it to ground fit and speed reasoning in them, never in the knowledge file's example specs. The analysis and summary model defaults resolve the same way — an explicit operator pick is honored, but a shipped default that a fresh box does not own falls back to a suitable model the box *actually* has (largest for Analyze's judgment, smallest for Summary's speed). The practical upshot: **one image, N boxes, change one URL.** Standing HELM up on a new machine requires pointing it at that machine's Lemonade and nothing else.

---

## 7. Storage and migrations

HELM's state lives in a SQLite database in the `/data` volume. SQLite is the right tool by design: HELM stores operator metadata and settings — low-volume, transactional, key/value-ish — not analytics.

`storage/migrations.py` follows a simple, strict discipline: a `schema_version` table, an ordered list of `(version, description, sql)` migrations applied in one transaction at startup, idempotent (safe to run every boot). The rules are **append-only** (never edit a shipped migration; add a new one) and **additive / never destructive without explicit operator action**. That second rule matters because of what the DB holds:

- **Operator model metadata** — descriptions, tags, favorites, archived state (migration 1). Lemonade tracks none of this.
- **Backend pids** — the crash detector's memory across polls (migration 2).
- **Advisor settings, HF metadata cache, saved summaries** — the editable instructions, the Layer 2 cache, and the persisted slow write-ups (migrations 5–9).
- **Warm set, benchmark results, config backups** — operator-triggered warm-on-startup, evidence for model choices, and pre-apply config snapshots (migrations 3–4).

**None of this is reconstructible from Lemonade.** A model's description, the operator's tags and favorites, a saved advisor summary, the record that a backend restarted twenty minutes ago — Lemonade has no memory of any of it. That is precisely why a careless migration would lose data with no upstream source of truth to rebuild from, and why the volume must be backed up. The JWT secret and operator password hash live here too (see §8).

---

## 8. Authentication

HELM ships its own operator login, on by default. A control plane needs its own gate for a blunt reason: HELM can unload the model that running agents depend on and delete gigabytes of weights. The spec as originally written had no auth; the implementation adds it because "read-only dashboard" and "can nuke the fleet" are the same URL away.

**Session cookie or bearer token.** The auth dependency reads a JWT from either the `helm_session` httpOnly cookie (browser) or an `Authorization: Bearer` header (scripts and agents — e.g. a warm-on-startup job). Sessions last 24h; API tokens last 90d. It is HS256 over a per-instance secret, single-operator (the subject is always `operator`; there is no user table and there should not be one). The secret resolves as *env override → persisted file → first-boot autogeneration* — no key ceremony to stand the app up, but the operator can take ownership for rotation.

**Auth holds by construction.** The dependency is mounted at the *router* level, never per-endpoint:

```python
_AUTH_REQUIRED = [Depends(require_auth)]
app.include_router(control.router, dependencies=_AUTH_REQUIRED)   # load / unload / pin / delete
app.include_router(library.router, dependencies=_AUTH_REQUIRED)
app.include_router(advisor.router, dependencies=_AUTH_REQUIRED)
```

On a control plane, a forgotten decorator on `POST /control/unload` is the whole ballgame — so auth cannot depend on the next developer remembering. Only two routers mount auth-open: `health` (for container healthchecks) and the `auth` endpoints themselves (they check credentials internally; `/auth/status` must answer before anyone has a token). First boot has no password: the UI detects the uninitialized state and prompts the operator to set one, and `/setup` refuses once initialized so it cannot be used to reset the password.

**The admin key stays server-side.** HELM's own login gates *HELM*. Lemonade's management routes are gated by a separate admin API key that lives only in the backend's environment and never leaves the `LemonadeClient` — the browser never talks to Lemonade, and nothing returns the key. If that key is unset, HELM runs read-only by design: the dashboard works, writes refuse with a clear message *before* the network call (a local "no admin key configured" beats a remote 401 the operator has to decode), and the UI hides the write controls. The startup log states which mode HELM booted in, so "management is disabled" is learned from the boot report, not from a button that 401s an hour later.

**The cookie's `Secure` flag is detected, not assumed.** Behind a TLS-terminating proxy, the browser is on HTTPS while the internal hop is plain HTTP, so neither `secure=True` always nor `request.url.scheme` is correct. `_detect_scheme` checks, in order: direct HTTPS, `X-Forwarded-Proto`, then the `Origin` and `Referer` headers (the load-bearing fallbacks for when a proxy forgets to forward `X-Forwarded-Proto`). A non-Secure cookie is logged loudly so a misconfigured proxy is visible rather than a silent login failure.

There is a documented escape hatch (`HELM_DISABLE_AUTH`) for the "mTLS at the edge is the only gate" case. It is an escape hatch, not a default: it logs a warning on *every single request*, so an unauthenticated control plane is impossible to leave switched on by accident.

---

## Design rules, in one place

- **An honest number or a visible `unknown` — never a plausible guess.** A control plane's confident wrong value is worse than no UI; the operator acts on it.
- **One parser / one source of truth, two-plus consumers.** Two hand-rolled implementations of the same read drift, and the drift is silent. `llamacpp_args`, `extract_pool`, and the concurrency allowlist each have exactly one definition.
- **Distinguish "absent" from "unknown" from "no".** `ABSENT` vs `MALFORMED`, `NO_LABELS` vs `UNKNOWN`, "search not run" vs "nothing found" — collapsing any of these manufactures a confident lie.
- **Ground, never recall.** The Advisor states model-specific facts only from provided context; the locked kernel makes that un-editable.
- **Detect the hardware, bake in nothing.** One image runs on any box.
- **Fail closed, and say why.** Auth by construction, secrets server-side, honest error classification, and a loud startup report over a silent 401.
