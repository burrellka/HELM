# HELM Control Plane Spec — Implementation Review

**Reviewing:** `C:\dev\Stris Halo Reaearch\14_HELM_CONTROL_PLANE_SPEC.md` v1.0
**Reviewer:** URSA-OSCAR dev (Claude) · **For:** H.A.L.O. (architect) + the HELM dev + Kevin
**Scope Kevin asked for:** what's wrong or hard to implement — especially §4's auth model and
§6's correctness rules — and anywhere HELM would silently diverge from how Vitals/URSA/APEX
already do things.

---

## Verdict up front

The spec is unusually good. It is grounded in real observed behavior, it names its traps
(`ctx_size` division, NPU exclusivity, the nuclear option, the 80GB saga), and §6 exists at all
— most specs never write down "here is what a wrong UI would do." The domain reasoning is
sound and I'm not second-guessing it.

The problems are **not** in the domain understanding. They're in three places:

1. **§4 specifies how HELM authenticates to Lemonade, and never specifies how the operator
   authenticates to HELM.** For a write-capable control plane, that's the blocking issue.
2. **Several §6 rules are specified against facts that aren't verified yet**, and a couple are
   written in a shape that will break the moment §5B.4 gets acted on.
3. **The spec is silent on the two mechanical things that will actually eat the dev's week**:
   streaming through NPM/Cloudflare, and per-operation HTTP timeouts.

Findings are ranked. F1–F6 I'd fix before the dev starts; F7–F11 before the phase that
depends on them.

---

# BLOCKING

## F1 — §4 has no operator→HELM auth at all. Only HELM→Lemonade.

**This is the finding that matters.**

§4's "Auth model (critical, and counterintuitive)" is entirely about the *upstream* leg:
`LEMONADE_API_KEY` vs `LEMONADE_ADMIN_API_KEY`, and keeping the key server-side. Rule #9 —
"The API key never reaches the browser" — is likewise about the Lemonade key.

**Nothing in the spec authenticates the human.** The only gate named is Cloudflare mTLS at
the edge (§2 "Ingress ... mTLS required").

Why that isn't sufficient:

- **The edge is not the only path in.** §2 puts HELM on `the Docker network` as a Dockge container.
  Anything on your LAN, any other container on `the Docker network`, and anything on the Docker host
  reaches HELM's port **directly** — NPM and Cloudflare are not in that path. mTLS protects
  the *public* route only. The LAN route is wide open by construction.
- **The blast radius is not "read some dashboards."** HELM as specced can unload the 26B that
  6 Hermes agents and 3 humans depend on (§1), `POST /api/v1/delete` 20GB of weights (§4.2),
  and rewrite global server config (§5B.3) including `enable_dgpu_gtt` — the one flag §5B.3
  itself calls "THE most consequential setting on the box."
- **We have already paid for this exact assumption.** URSA 1.1.9 was a CRITICAL security fix:
  we shipped an MCP server whose OAuth provider auto-approved, and the only thing between the
  internet and the operator's medical data was an assumption about the edge. It cost a
  same-day emergency release plus secret rotation. The lesson we wrote down was *defense in
  depth: never let the edge be the only gate.* HELM as specced repeats the structure.

**House conflict:** URSA and Vitals both run app-level operator auth — single-operator
Argon2id password → HS256 JWT → httpOnly cookie — **in addition to** edge protection. HELM
would be the first app in the family with no auth of its own, and it's the one that can
delete things.

**Recommendation:** take URSA's auth module. It's already ported into the scaffold at
`backend/src/kblocallm/auth/` — password setup on first boot, login, JWT cookie, and the
`_AUTH_REQUIRED` include_router pattern (see F1a). It is roughly a day of the dev's time,
most of which is already written.

If H.A.L.O. and Kevin decide mTLS-only is acceptable, that's a legitimate call — but it needs
to be **an explicit written decision in the spec**, plus: bind the container port to
`127.0.0.1` only (not `0.0.0.0`), and document the LAN exposure so nobody later assumes
there's a login. What I'm objecting to is the silent version.

### F1a — while you're there: auth by construction, not by discipline

URSA applies auth at the router-mount level, not per-endpoint:

```python
_AUTH_REQUIRED = [Depends(require_auth)]
app.include_router(models.router,  dependencies=_AUTH_REQUIRED)
app.include_router(control.router, dependencies=_AUTH_REQUIRED)
```

The comment in `ursa_oscar/main.py` explains why, and it applies double to HELM: *"a developer
who adds a new endpoint to an existing protected router can't forget to auth-guard it."* On a
control plane where a forgotten decorator means an open `POST /delete`, per-endpoint auth is
the wrong default. The scaffold sets this up already.

## F2 — §5.1 and Rule 4 hardcode `120.5`, which is the bug Rule 4 exists to prevent

Rule 4 says the pool "is 120.5 GB, not 80," and that HELM "must never reintroduce that
error." Then §5.1 specifies the header as:

> **VRAM: `vram_gb` / 120.5 GB** with a bar

`120.5` as a literal is the same class of bug as the 80GB one, just pre-loaded with today's
right answer. The 80GB figure was *also* correct-looking until the thing underneath it
changed. Concretely, this constant goes wrong when:

- `enable_dgpu_gtt` is ever `false` (§5B.3 says it ships `false`, and HELM's own config editor
  can set it) — the real pool becomes 0.5GB and HELM cheerfully draws a bar against 120.5;
- the box gets more RAM, or HELM is ever pointed at a second machine.

**Fix:** read the pool from the API on every poll, render it, and cross-check it against
`enable_dgpu_gtt`. Hard-warn when the reported pool is small *and* `enable_dgpu_gtt` is false —
that's the 80GB saga's actual signature, and HELM is well placed to catch it in one glance
instead of months. Never let `120.5` appear in the source except as a test fixture.

**And the field is unverified.** Rule 4 says read `virtual_mem_gb` from `/api/v1/system-info`,
but §4.1 describes that endpoint as returning "CPU/GPU/NPU hardware specs, `virtual_mem_gb`,
OS". On Linux "virtual memory" conventionally means RAM+swap — *not* the GTT pool. It may
well be the right field here, but the doc asserts it without a captured response, and Rule 4's
entire job is to stop a wrong number. **Verify against the live box and paste the actual JSON
into the spec before this becomes a constant.** If `virtual_mem_gb` turns out to be system RAM,
Rule 4 currently instructs the dev to reintroduce a variant of the exact bug it's forbidding.

Related: §5.5 says `vram_gb` is "a **total**", meaning total *in use*; §5.1 uses it as the
numerator against a capacity denominator. Those are consistent but the doc never says
"`vram_gb` = current usage, `virtual_mem_gb` = capacity" in one place. Say it once, explicitly.

## F3 — Rule 2 is a denylist, and §5B.4 actively breaks it

Rule 2: *"FLM/NPU models have NO slots. Show `SERIAL`. Never fabricate a slot count from
`ctx_size` when `recipe == "flm"`."*

That's written as: *if flm → SERIAL, else → slots*. But §5B.4 and §5B.1 enthusiastically
push installing **whispercpp, sdcpp, kokoro, moonshine** — and §3's own schema lists
`recipe: "llamacpp" | "flm" | "whispercpp" | "sdcpp" | ...`. A Whisper backend is neither
`llamacpp` nor `flm`, so it falls into the `else` branch and HELM renders a **fabricated slot
count** for it — precisely the "confidently mislead" failure Rule 2 exists to prevent. The
spec breaks its own rule two sections later.

**Fix — invert to an allowlist:** slots are a *llamacpp* concept.

```
recipe == "llamacpp"  → derive and show slots
everything else       → SERIAL / not-applicable
```

That's correct for flm today, correct for whisper/sd/kokoro tomorrow, and correct for the next
recipe nobody's thought of. Rewrite Rule 2 in the allowlist form.

## F4 — the `--parallel` parser is the highest-risk function in the product, and Rule 1 rides on it

Rule 1 and §4.1 specify:

```
parallel = parse("--parallel N") from llamacpp_args, default 1
per_slot = ctx_size / parallel
```

The failure mode if the parse misses is **silent and severe**: it falls back to `default 1`,
so `per_slot = ctx_size = 524288`, and HELM displays **8× the real context** with total
confidence. That is the exact "worse than no UI" scenario §6 opens with.

And it will miss, because `llamacpp_args` is a **free-text string the operator can edit**
(§5.3 "Raw args ... escape hatch"), and llama.cpp accepts more than one spelling:

| Form | Naive `--parallel N` regex |
|---|---|
| `--parallel 8` | ✅ |
| `--parallel=8` | ❌ → falls back to 1 → **8× wrong** |
| `-np 8` (llama.cpp short form) | ❌ → **8× wrong** |
| `--parallel  8` (double space) | ⚠️ depends |
| `--parallel 8 --parallel 4` (dup) | undefined — which wins? |
| `--parallel abc` | ⚠️ must not crash the dashboard |

**Fix:** one shared, tested `llamacpp_args` parser module. Not a regex inline in a view.
Handle long/short/equals forms, last-wins on duplicates, and a malformed value must degrade to
a visible "unknown" — **never to a plausible number**. Table-driven unit test over every form
above.

**And make it the same parser the reserved-arg validator uses.** §5.3's validation list has
the identical problem: it must reject `--ctx-size`, but also `-c`, and `--ctx-size=4096`. Two
independent hand-rolled arg parsers in one codebase will drift. One parser, one test table,
two consumers.

> This is the same bug class we just shipped a fix for in URSA 1.1.15: the AI tool schema and
> the metric validator were two hand-maintained lists that disagreed, the model guessed a
> plausible-but-wrong value, and the failure was silent. The structural fix — derive from one
> source of truth, add a drift test — applies verbatim here.

## F5 — `/api/v1/logs/stream` and `pull stream:true` through NPM + Cloudflare will 524. The spec doesn't mention streaming at all.

§4.1 specifies a streaming log tail; §5.6 makes it a feature; §4.2/§5.5 use `pull` with
`stream: true`. §2 puts **Cloudflare + NPM** in front. The spec says nothing about how a
stream survives that path — and by default, it doesn't:

- **Cloudflare kills a response with ~100s of no bytes** (error 524). It is not a limit on
  total duration — it's an *idle* timer. A quiet log stream (no new lines for two minutes at
  3am) is exactly that.
- **NPM's nginx buffers responses by default**, so the stream never flushes to the edge at
  all. The symptom is maddening: the backend is streaming fine, the UI shows nothing, then it
  dies.

**This is a solved problem in the family** — it's the whole subject of
`C:\dev\fitbit-web-ui-app\docs\sse-streaming-for-sibling-devs.md`, and I implemented it in
URSA (`frontend/nginx.conf` + `api/ai.py`). Give the dev that note. The short version:

1. FastAPI returns a real `StreamingResponse` / `EventSourceResponse` (not a buffered body),
   with `Cache-Control: no-cache` and **`X-Accel-Buffering: no`**.
2. A **keepalive comment every ~15s** on quiet streams (URSA uses `: keepalive\n\n` at 5s).
   This one line is what defeats 524.
3. **NPM specifically:** `proxy_buffering off` + `proxy_http_version 1.1` + a long
   `proxy_read_timeout` must go in the proxy host's **"Custom Nginx Configuration"** box —
   NPM's UI has no checkbox for it, which is exactly why this gets missed. Ingress notes with
   a paste-ready block are in the scaffold at `infra/npm-mtls-ingress.md`.
4. Frontend consumes with `fetch` + `ReadableStream` (not `EventSource` — it can't POST and
   can't send auth headers).

## F6 — one HTTP timeout will not fit HELM's operations

Not mentioned in the spec, and it will bite in week one. HELM→Lemonade operations differ by
**four orders of magnitude**:

| Operation | Realistic duration |
|---|---|
| `GET /live` | milliseconds |
| `GET /health`, `/system-stats` | milliseconds (polled every 2s) |
| `POST /load` | **10–20s** (the spec's own number in §5.3) |
| `POST /pull` | minutes to hours |
| `GET /logs/stream` | unbounded by design |

A single global client timeout either kills every pull or makes a dead health check hang the
dashboard. Worse, **httpx has four timeouts** (connect/read/write/pool); `timeout=30.0` sets
all four, but a `Timeout(...)` object with only some fields set silently leaves `read` at the
5s default. We lost real time to that in URSA before 1.1.11.

**Fix:** per-operation timeout profiles, explicit about all four legs. The scaffold ships this
as `upstream/client.py` with named profiles (`FAST`, `CONTROL`, `TRANSFER`, `STREAM`) so the
dev picks an intent, not a number.

---

# SIGNIFICANT

## F7 — Rule 5's "warm on startup" is a write action triggered by an inferred state. It's the most dangerous item in §6.

The gap it identifies is real and worth fixing (pinned ≠ autoload; after `systemctl restart
lemond`, everything is cold). But as specced, HELM would **issue loads based on inferring that
lemond restarted**, and the spec gives it no reliable signal to infer from.

The obvious inference — `all_models_loaded == []` — is **the same observation as §3's
"nuclear option"**, where a backend failure makes the router deliberately evict *everything*
to clear hardware state before retrying. So HELM would watch the router frantically clearing
hardware, conclude "looks like a restart!", and start mass-loading models into it. That's not
a race condition, it's HELM fighting the router during an incident.

Also unaddressed: two HELM instances (or a browser tab and a scheduled job) both warming;
warming an NPU model evicting the incumbent (Rule 7) with no human present to see the warning.

**Recommendation:** keep it in Phase 4, and require all of:
- an explicit operator-defined **warm set** (not "everything that was pinned"), default empty;
- a real restart signal, not an empty list — verify whether the router exposes an uptime,
  start time, or a monotonic that resets (`last_use` is monotonic; `websocket_port`/`version`
  won't help). **If no such signal exists, say so in the spec** — it changes the design;
- single-flight lock + debounce (don't warm within N seconds of a mass-eviction event seen in
  the log stream);
- **never auto-warm an NPU model** — exclusivity means it evicts, unattended;
- default **off**.

## F8 — Rule 8: is `watchdog_reset` sticky? A permanent ⚠ is a warning nobody reads.

§3 says it "tells you whether that backend has **ever** been force-restarted." If it never
clears, then one crash three weeks ago paints a permanent ⚠ on the row, and within a week the
operator has learned to ignore the warning triangle — which costs you the *next* incident.

The spec doesn't say whether it resets on unload/reload. **Verify.** Then either way HELM
needs: a **timestamp** ("crashed 3d ago") and an **acknowledge** action stored in HELM's own DB
(you already have a store for operator metadata per §5.3a — same place). Surface *unacked*
incidents loudly; keep acked ones visible but quiet.

## F9 — Rule 10 depends on a join the spec never verifies

Rule 10 — "Trust `labels`, not model names" — is right, and the reasoning (FLM silently strips
`tools` for models without the `tool-calling` label) is the best argument in the document.

But `labels` come from `GET /api/v1/models?show_all=true` keyed by **`id`**, while loaded state
comes from `GET /api/v1/health` keyed by **`model_name`**. HELM must **join** them, and the
spec's own two examples don't obviously share a namespace:

- health: `"model_name": "Gemma-4-26B-A4B-it-MTP-GGUF"`
- models: `"id": "gemma4-it-e2b-FLM"`

They're probably the same key space. "Probably" is doing a lot of work for the rule that
prevents an agent being wired to a model that silently hallucinates instead of calling tools.
**Verify the join key against the live box and write it into the spec.**

And specify the miss behavior: an unmatched join must render **"capabilities unknown"**, never
an empty badge row. Absence of evidence isn't evidence of absence — a model showing *no*
`tool-calling` badge because the join failed looks identical to one that genuinely lacks it,
and Rule 10's whole point is that the operator acts on that badge.

## F10 — §5.5's delta memory attribution is racier than it reads

I agree with the spec's ranking (option 1 over option 2) and especially with "do not fabricate
precision." Two things to add:

**The delta is not clean.** Snapshot `vram_gb`, load X, snapshot again → attribute the delta
to X. But: LRU eviction is **asynchronous and triggered by that very load**, so the delta can
include a *negative* contribution from something the router evicted to make room. Other
clients (the CLI, KAIROS itself, a second HELM tab) load concurrently. The nuclear option can
fire mid-measurement. Any of those produce a confidently wrong per-model number — the failure
mode §5.5 is explicitly trying to avoid.

**Fix:** single-flight lock around measured loads; capture the full `all_models_loaded`
composition before/after and **invalidate the measurement** if anything changed besides the
target appearing; keep the "estimated" marker; show nothing rather than a number you know is
polluted.

**And strengthen the case against option 2.** `pid: host` doesn't just "couple HELM to the
host" — it hands the container the host PID namespace, i.e. visibility into and signal
capability against every process . On a control plane that (per F1) currently has
no operator auth, that's not a coupling trade-off, it's a privilege escalation waiting for a
bad day. Reject it on those grounds, not just tidiness.

## F11 — `?include_paths=true` (§4.1) is a data leak through an unauthenticated UI

The spec already flags that it "leaks local usernames/cache layout" and says to request it
deliberately. Reinforce: **HELM's backend must not pass it through by default.** Make it an
explicit, per-request operator action. Combined with F1 (no auth), a casual proxy of this
endpoint publishes the host's filesystem layout to anything that can reach the port.

---

# MINOR

- **§5.5 is used twice** — "5.5 Download manager" and "5.5 ⚠️ THE MEMORY-ATTRIBUTION GAP".
  Renumber; the dev will cite these.
- **Rule 3 (`npu_percent`), TypeScript detail:** use `npu_percent ?? 'unavailable'`, **not**
  `||`. If the gauge is ever fixed upstream and legitimately reports `0`, `||` renders a true
  0% as "unavailable" — a new lie in the slot where you removed the old one.
- **Polling cadence.** §7 Phase 1 polls `/health` + `/system-stats` every 2s per open tab.
  The spec already notes `/live` "does no work" — use it: `/live` at 2s for liveness,
  `/health` at ~5s, and let the log stream carry events. Free, and it keeps the dashboard from
  being a load source on the box it's monitoring.
- **HELM's own datastore is never located.** §5.3a says "SQLite or a JSON sidecar", §2 says
  data lives at a host data directory. Pin it: SQLite at `/data/helm.db`, bind-mounted
  from a host data directory. The scaffold does this.
- **How the admin key reaches the container isn't specced.** §8 says it's in
  the Lemonade server config on the host; HELM is a container. Say: env var via
  the Dockge stack's `.env`, `${LEMONADE_ADMIN_API_KEY}` in compose, **never baked into the
  image**, never in the repo. Scaffold has `.env.example` + compose wiring.
- **Missing-key behavior.** If the admin key is absent, every write 401s with no explanation —
  the operator debugs for an hour. URSA 1.1.8 fixed this exact experience with a loud startup
  banner. HELM should: fail loudly at boot with an actionable message, and have the UI render
  **"management disabled — no admin key configured"** with writes hidden, rather than
  presenting buttons that mysteriously fail.

---

# HOUSE-STYLE CONFLICTS (flagging rather than diverging silently)

Kevin's ask. Ordered by consequence.

| # | Vitals / URSA does | HELM spec says | Recommendation |
|---|---|---|---|
| 1 | **App-level operator auth** (Argon2id → JWT → httpOnly cookie) **plus** edge protection | Edge mTLS only; §4 auth is upstream-only | **Adopt.** See F1. Biggest gap. |
| 2 | **Auth by construction** — `_AUTH_REQUIRED` at `include_router` | not addressed | Adopt (F1a). Matters more here than in URSA. |
| 3 | **Same-origin**: one nginx serves the SPA *and* proxies `/api` → no CORS at all | §4 mentions Lemonade's CORS; HELM's own posture unstated | Same-origin. Don't enable CORS on HELM; there's no cross-origin caller. |
| 4 | **Secrets: env > persisted file > first-boot autogen**, no key ceremony | not addressed | Adopt for HELM's JWT secret. Scaffold has it. |
| 5 | **Never collapse distinct failures into one message** (the "AI unavailable" lesson) | not addressed | HELM's writes fail distinctly: 401 (no admin key) ≠ 409-ish (NPU exclusivity evict) ≠ 500 (backend crash) ≠ timeout. Show which. |
| 6 | **Version self-introspection** (container reports its own version; no env-var sync) | not addressed | Adopt; it's ~20 lines and it ends "which image is actually running" forever. |
| 7 | **nginx `resolver` + `proxy_pass` via variable** | not addressed | Required on `the Docker network` — without it nginx caches a container IP at config load and a restart silently breaks the proxy. We hit this in APEX. |
| 8 | **DuckDB** (URSA/Vitals) | SQLite / JSON sidecar (§5.3a) | **No conflict — SQLite is right here.** DuckDB is for analytics; HELM stores operator metadata and settings. Deliberate, documented divergence. |

---

# WHAT I'D CHANGE IN THE SPEC BEFORE THE DEV STARTS

Small edits, high leverage:

1. **Add §4.0 "Operator auth to HELM"** — resolve F1 one way or the other, in writing.
2. **Rewrite Rule 2 as an allowlist** (`recipe == "llamacpp"` → slots; else SERIAL). F3.
3. **Delete the literal `120.5`** from §5.1 and Rule 4; specify "read from the API, cross-check
   `enable_dgpu_gtt`." Paste a real `/api/v1/system-info` response to settle which field is
   the pool. F2.
4. **Add a "verify before building" list** — the spec is honest that some things need
   confirming (it already does this for pin, §4.2). Add: the `health`↔`models` join key (F9),
   whether `watchdog_reset` clears (F8), whether any restart signal exists (F7), and the pool
   field (F2).
5. **Add a §4.4 "Streaming and timeouts"** — the 524/buffering rules and the per-operation
   timeout table. F5, F6.
6. **Rule 1 → "use the shared args parser"**, and note the reserved-arg validator uses the
   same one. F4.

None of that changes the architecture. It's the difference between a dev shipping this in
three weeks and a dev discovering §F5 at 11pm behind Cloudflare.

---

# WHAT'S GOOD (so it doesn't get edited out)

Worth saying plainly, because review docs skew negative:

- **§6 existing at all.** "A UI that gets these wrong is worse than no UI" is the right
  standard, and every rule traces to a real incident.
- **The `ctx_size` trap and the derived-per-slot rule.** That's a genuine insight and it's the
  reason HELM will be better than the thing it replaces.
- **The `labels` / `tool-calling` reasoning (§4.3).** The observation that FLM silently strips
  tools for unlabeled models — and that the label is the *only* reliable signal — is the single
  most valuable paragraph in the spec. It's a correctness argument, not a UI preference.
- **SAVED vs RUNNING drift (§5.3).** Nobody has this. It's the best feature in the product and
  it falls straight out of data already on the wire.
- **Refusing to fabricate** — `npu_percent` "unavailable" not 0%, memory "estimated" not
  precise, `SERIAL` not a fake slot count. That instinct is why this spec is trustworthy.
- **Phase 1 = read-only.** Correct. Ship the dashboard, earn the write surface.
