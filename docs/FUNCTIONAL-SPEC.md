# HELM — Functional Specification

HELM is a self-hosted web control plane for a single Lemonade Server instance
(a local-LLM runtime that hosts multiple models behind one OpenAI-compatible
API). Lemonade already exposes the raw endpoints; HELM is the layer that turns
them into something an operator can trust under pressure.

This document describes **what each surface does** and, for every behavior, the
**correctness rule** that makes the surface trustworthy. HELM's premise is that a
control plane which reports a confident wrong number is worse than no control
plane at all — a wrong memory ceiling once cost this project months. So the
recurring theme below is not "show data"; it is "show only what can be stood
behind, and mark everything else honestly as unknown."

## Design invariants (they recur on every surface)

These principles are implemented once, in tested pure-Python domain modules, and
consumed by the API and UI. They are not per-screen conventions; they are the
product.

- **Honest ignorance beats a confident guess.** Any value HELM cannot read
  renders as `unknown` / `unavailable` / `no measurement` with a distinct visual
  treatment — never as `0`, never as a remembered constant, never as a plausible
  default. A blank where a number belongs is a feature.
- **Derived numbers are decided server-side, once.** The frontend receives
  already-decided values and a display string; it never divides, multiplies, or
  re-parses. Two implementations of the same rule drift, and the drift is
  silent. There is exactly one arg parser, one slot-math function, one pool
  extractor.
- **Keep the KIND of a failure.** "Unreachable", "unauthorized", "timeout", and
  "backend dead" are four different operator actions. They are never collapsed
  into a generic "error."
- **Partial degradation, not blank screens.** Each upstream read fails into its
  own section. If one of several calls times out, every section that succeeded
  still renders — the operator is usually looking at HELM precisely because
  something is already wrong.
- **Portability.** Nothing about a specific machine is baked into code. Hardware,
  pool size, and available backends are all detected live, so HELM deployed to a
  different box reports that box.

---

## 1. Dashboard (read-only)

The dashboard is the always-on situational view: per-model status, the real
memory pool, derived per-slot concurrency and context, throughput, and crash
indicators. It performs no writes. A liveness poll runs on a tight interval and
is kept separate from the heavier assembly poll, so the UI can distinguish "the
box is up but one read failed" from "the box is gone" — those demand different
reactions. Near-static reads (hardware info, server config) are cached and
refreshed rarely so the dashboard never becomes a load source on the box it
watches; the last good value is served while a refresh is briefly failing,
because the number did not change — HELM just failed to re-read it.

### Behavior

- **Per-model rows**: name, checkpoint, device, recipe/backend label, concurrency
  (see below), pinned flag, evaluated status, and an LRU column.
- **Memory pool bar**: current GPU memory use as a percentage of the *real* pool
  capacity, with the exact field path it was read from shown on hover.
- **Per-type capacity**: loaded/max/pinned counts, always **per model type**,
  never a single global cap.
- **Throughput**: tokens/sec, time-to-first-token, and input/output token counts
  for the last request, plus lifetime totals.
- **Status and restart indicators**: a per-model READY / NOT_READY / DEAD state,
  and a restart notice when a backend's process id changed.

### Correctness rules

- **The 8x-lie guard (per-slot context).** Lemonade's `ctx_size` is the *total*
  context divided across parallel slots, not the per-slot value. A model reading
  `ctx_size: 524288` with 8 slots is 8 × 64K, not 8 × 512K. The dashboard only
  ever shows the derived per-slot number. Crucially, if the slot count cannot be
  read from the free-text args string, the concurrency cell renders `unknown` —
  it must **never** fall back to "1 slot", because that would render the total as
  the per-slot value: eight times the truth, stated with total confidence. This
  is the single highest-risk computation in the product, which is why the arg
  parser refuses to guess and has a status distinct from "absent."
- **The pool denominator is read live and cross-checked.** The upstream payload
  carries two numbers side by side — the real unified memory pool and a tiny BIOS
  carve-out — and the wrong one has the friendlier field name. HELM reads the
  pool per poll from a single tested extractor, shows the field path it used, and
  will render `unknown pool` rather than fall back to the carve-out. When the pool
  reads implausibly small *and* the GTT-enable flag is off, it raises the exact
  signature of the historical memory saga with an actionable fix, instead of
  silently drawing a bar against a phantom ceiling.
- **NPU utilization renders `unavailable`, not `0%`.** The upstream NPU gauge is a
  known-broken null on this class of hardware. The UI uses null-coalescing
  (`?? 'unavailable'`), never a truthy-or (`|| 0`): if the gauge is ever fixed and
  legitimately reports a true `0%`, a truthy-or would render that as
  "unavailable" — swapping one lie for another. Genuinely-unavailable is a
  different claim than "idle," and the two never collapse.
- **Implausible throughput is shown *and* flagged.** After a one-token response,
  the upstream can report a rate like 1,000,000 tok/s — the artifact of dividing
  a tiny token count by a near-zero elapsed time. Above a plausibility ceiling,
  HELM shows the reported number but labels it "no measurement," so it is never
  passed off as real throughput.
- **The LRU column shows rank, not a fabricated duration.** The usage clock is a
  monotonic counter with no epoch and no wall-clock anchor anywhere in the API, so
  "3 minutes ago" would be invented. HELM shows *rank* (1 = most recent) and marks
  the single unpinned model the router would evict first — which is the actual
  question the column exists to answer. Pinned models are excluded from the
  eviction marker, including a pinned model with the oldest usage, exactly the row
  a naive "oldest wins" would wrongly condemn.
- **Restart, not "crash."** A process-id change proves a backend restarted; it
  does **not** prove a crash, because the operator running a load at the CLI
  produces identical evidence. HELM says "restarted" — what it can see — and only
  escalates to "crash" when the watchdog flag corroborates. That flag is
  corroboration only, never the trigger: it stays false when a request lazily
  reloads a dead backend before the watchdog polls, which is the *normal* case on
  a busy model. A recent-restart window ages the notice out of urgency on its own,
  so a permanent warning never trains the operator to ignore warnings.
- **A dead backend still reads DEAD.** If HELM's own state store is unavailable,
  the fallback reads liveness straight off the record rather than inheriting a
  stale cheerful "ready" the router left behind.

---

## 2. Control (write)

The control surface performs load, unload, and pin, plus per-model settings:
slots, per-slot context, backend, and structured sampling parameters
(temperature, top-k, top-p, min-p, repeat-penalty). Every write is planned by a
pure, tested function before any network call; the plan is auditable in the UI as
the exact body that would be sent. Anything with a consequence the operator
cannot un-ring requires an explicit second confirmation: the first call returns
the plan and performs nothing, the UI shows the warnings, the second call carries
the confirmation. A control plane should not turn a click into an eviction with
no step in between.

### Behavior

- **Load** with structured fields, a dry-run preview endpoint the dialog calls as
  the operator edits, and a confirm gate for consequential loads.
- **Unload**, with warnings that name the blast radius (in-flight requests fail;
  a reload costs seconds; it does not come back after a server restart).
- **Pin / unpin** as a metadata flag with no model reload.
- **Per-model settings** entered as labeled fields, not raw text.

### Correctness rules

- **The operator never does the context multiplication.** They think in "8 slots
  of 64K"; Lemonade wants the total (`ctx_size: 524288`). HELM does the
  multiplication and shows the computed total as a *consequence* of the request,
  never as an input the operator must compute. This asymmetry is the entire
  reason the structured editor exists — the raw total is the number that misled
  everyone.
- **Reserved args are rejected before send.** A set of arguments (context size,
  vision projector, draft model, device placement, port, and others) are how
  Lemonade auto-wires its own behavior; it hard-errors on them. HELM catches them
  in the *same* parser the slot math uses, so the raw-args escape hatch cannot
  smuggle in a flag the structured form would have blocked. Raw editing is more
  powerful, not more correct.
- **A value set in two places is refused.** If `--parallel` (or any sampling
  flag) appears both in its own field *and* in the raw-args box, HELM blocks the
  load. Two sources disagree silently depending on Lemonade's arg order; refusing
  is the only honest option. The fields own their flags, and what remains in the
  raw box is the genuine escape hatch.
- **NPU exclusivity is warned by name, before the click.** Only one NPU backend
  may be active at a time — a router-level rule, not a preference. Loading a
  second NPU model silently unloads the first, which unattended is an agent losing
  its brain mid-conversation. The plan names *which* model will be unloaded and
  requires confirmation. Slots are rejected for any non-slotting recipe, because
  slots are a single-recipe concept and fabricating a slot count for the rest was
  a real historical bug.
- **HELM does not fake a memory estimate.** The spec asked for a pre-commit
  footprint estimate; Lemonade exposes no per-model memory, so the footprint of a
  never-loaded model is genuinely unknown. HELM states the *known* headroom and
  the eviction consequence, and declines to invent a "needs ~14GB" number.
- **Every write refuses honestly without an admin key.** Management endpoints
  check for the admin key and return a clear local error *before* the network
  call — a plain "no admin key configured, set it and restart" beats a remote 401
  the operator has to interpret. Read-only mode is a first-class supported state:
  when no key is present, the UI hides write controls and says why rather than
  rendering buttons that fail. HELM-local metadata (descriptions, tags,
  favorites) is deliberately *not* gated on the key, so a read-only HELM is still
  the operator's notebook.
- **Control action timeouts use the reload budget, not the fast one.** A reload
  takes real seconds; reporting a timeout for a load that actually succeeded is
  the worst lie a control plane can tell, because the operator retries and the
  second load evicts something.
- **Pin is verified by read-back.** The pin route body shape was reconstructed
  from a packet capture, so HELM does not take the success response at its word:
  it re-reads health and confirms the flag actually flipped. If it did not, HELM
  reports that plainly instead of a false "done." An already-in-state pin reports
  as done rather than making a no-op call that could look like a failure.

---

## 3. Library

The library is the model catalog joined with live loaded-state and HELM's own
operator metadata. It supports a unified search across every axis at once
(text, capability labels, tags, recipes, downloaded/loaded/favorite), downloads,
deletes, and per-model notes. The catalog is cached with a short TTL and
invalidated explicitly on pull and delete.

### Behavior

- **Catalog rows** with capability badges, size, trained context ceiling, and
  downloaded/loaded/pinned state.
- **Downloads** are server-owned: the pull job is handed to Lemonade and polled,
  so a large download survives a browser refresh. HELM does not proxy the
  download stream through itself (that would tie the job's life to one HTTP
  connection through the ingress path and time out).
- **Deletes** require an explicit confirm and refuse a currently-loaded model.
- **Operator metadata** — descriptions, tags, favorites, archive — is HELM-local,
  exportable, and importable as an additive merge.
- **Saved-vs-running settings** shown side by side per model.

### Correctness rules

- **The capability join is tri-state, never collapsed.** A model can be
  **matched** (joined, publishes labels), **no-labels** (joined, genuinely
  publishes none), or **unknown** (the join failed). These are three distinct,
  visually-different states. An unmatched model renders "capabilities unknown," and
  loaded models with no catalog row are surfaced explicitly — never an empty badge
  row. Absence of evidence is not evidence of absence: a failed join looks
  identical to a genuine no-capability model, and conflating them would let HELM
  vouch for a model it knows nothing about.
- **Silent tool-stripping is warned — but only where it can bite.** On the NPU,
  tool-calling is compiled per-model into the kernel; the runtime **silently
  strips the tools array** for a model that lacks it, so the model never sees the
  tools and hallucinates an answer. The capability label is the only reliable
  indicator — not the name, not the size, not the family. HELM flags
  "strips tools silently" only when the join actually succeeded (a failed join is
  "unknown," not a false accusation) and only for chat-capable NPU models — it
  does not fire on an embedding or reranking model, where a tool call is not a
  thing anyone would ask. A warning that fires where it cannot matter is how the
  warning that *does* matter gets ignored.
- **Provenance labels are not capability badges.** Curation/provenance markers
  are kept out of the capability badge set, so they never dilute the one badge
  that prevents silent failure.
- **Saved-vs-running drift is honest about the benign case.** A model can run with
  options that were never persisted; when the server restarts, the running config
  silently reverts to whatever was saved — possibly nothing. HELM diffs saved
  against running and separates real disagreement (`DRIFT`) from
  "running but never saved" (`NOT_SAVED`, a restart loses it) from "saved but not
  in effect." Fields Lemonade auto-detects (the backend, present in running and
  absent in saved on every row) are marked benign, so the drift indicator is not
  always-on — an always-on indicator is one nobody reads.
- **Downloads keep the operator's chosen name.** A custom pull that should use a
  namespaced name is *warned*, not silently rewritten — renaming what the operator
  typed is its own surprise.
- **The catalog listing does not leak the host filesystem.** The path-listing
  option on the upstream models endpoint is deliberately not passed; a control
  plane should not publish the host's directory layout as a side effect of listing
  models. That remains an explicit, separate operator action.

---

## 4. Model Advisor

The Advisor helps a non-expert operator understand a model: what its
quantization and size mean, whether it fits the hardware, what it is good for,
and which owned model might be a better choice. It is built in **three provenance
layers**, and every field it returns carries its layer so the UI can label them
distinctly.

- **Layer 1 — local facts.** Deterministic arithmetic and string parsing over the
  catalog: decode the quant, count parameters, compute the footprint, check the
  fit against the real pool. **No model runs**, so this layer cannot hallucinate.
  It renders instantly and is the floor the other layers cannot contradict.
- **Layer 2 — external metadata.** A HuggingFace card, fetched only when the
  model's checkpoint carries an explicit `org/repo` id. HELM never fuzzy-matches a
  name to a plausible repo — a wrong guess attaches the wrong model's card and
  grounds the whole advisor on someone else's model. Anything ambiguous, a local
  path, or a bare tag resolves to "no external record," backed by Layer 1 alone.
- **Layer 3 — synthesis.** The analysis model generates fresh text over Layers 1
  and 2, clearly labeled as generated and "not an independent source." On any
  failure it returns a labeled "unavailable," never a fabricated summary.

### Behavior

- **Analyze / deep synthesis** — the heavy teaching path: full facts card plus a
  thorough, beginner-friendly written analysis grounded on the operator's actual
  inventory and detected hardware.
- **Summary** — a deliberately fast, cheap read: its own small model, its own
  instruction box, a starved context (the facts card only — no inventory, no web
  search, no knowledge file), and a low token cap. Targets a couple of seconds.
- **Chat about a model** — a streaming conversation anchored to the model's Layer
  1 facts and Layer 2 card, searching on the operator's actual question.
- **Settings** — editable analysis model, summary model, instruction boxes, and
  knowledge file, plus a model picker that flags sub-2B models as unsuitable
  analysts.
- **Saved summaries** — persisted as dated snapshots and shown instantly.

### Correctness rules

- **Grounded, never recalled.** Every generated layer is told to answer only from
  the provided context and to say "unknown" for anything the context does not
  contain. Local facts are labeled facts; external metadata is labeled external;
  generated text is labeled generated. The UI keeps the three visually separate so
  the operator always knows which is which.
- **A locked safety floor no edit can disable.** The operator can rewrite the
  analysis instructions, the summary instructions, and the knowledge file. The
  "do not fabricate" safety kernel is a code constant, not an editable field — no
  instruction edit can turn it off. The settings endpoint reports the operational
  prompt as locked.
- **Summary is a different tool, not a lighter synthesis.** It builds its own tiny
  context, runs its own model, and caps tokens low — so a small non-reasoning
  model is the *right* choice here and is recommended, whereas the same model is
  flagged as an unsuitable *analyst* elsewhere. The sub-2B "fabricates" warning
  applies to the analyst role and is deliberately dropped for the summary role.
- **Web search is a grounded source, honestly marked when absent.** When a search
  key is configured, the advisor runs a real grounded lookup and cites its
  sources; the key lives in the environment and is never returned to the client,
  while the provider and model are operator-editable so a newer grounding model
  takes effect with no redeploy. When no key is set, search is marked "not run"
  and the model is told to say so rather than fill the gap from memory. A search
  failure comes back as a structured result the model can reason about, never a
  silent blank.
- **The reasoning-token starvation trap is surfaced, not swallowed.** A reasoning
  model spends its token budget on a hidden reasoning channel *before* emitting
  the answer; a tight budget starves the answer to empty. HELM reads the content
  channel (never the reasoning scratchpad — surfacing that would be its own lie),
  detects the "reasoned but produced nothing, truncated at length" case, and
  reports it honestly with a pointer to pick a non-reasoning model — rather than
  showing a blank. The streaming chat relays the reasoning channel as a live
  "thinking" indicator, distinct from the answer, so the operator sees motion
  during the long first-token wait.
- **Saved summaries are dated snapshots, not live reads.** A saved summary carries
  its date and the model that wrote it, and is presented as a snapshot so it is
  never mistaken for a current answer. Re-generating overwrites it.
- **Portable model resolution.** An explicit operator model pick is honored as-is
  and fails honestly if absent — they chose it. Only the shipped *default* falls
  back: Analyze falls to the largest suitable owned model (for judgment), Summary
  falls to the smallest (for speed), so HELM on a fresh box still works before the
  operator configures anything.

---

## 5. Operations

Operations is where HELM does live, consequential things carefully: a log
stream, warm-on-startup, benchmarks, and a live system panel.

### Behavior

- **Live log stream** — the raw upstream log tail, kept raw on the wire so the
  pane shows exactly what the server said, with classification (highlighting)
  applied by the same tested rules the server exposes.
- **Warm set + warm** — an explicit list of models to bring back after a restart,
  and an operator-triggered button to load them.
- **Benchmarks** — run a fixed scenario against a loaded model and store the
  result with its configuration, so model choices are evidence-based.
- **Current system setup panel** — the live, detected hardware of the box HELM is
  pointed at.

### Correctness rules

- **Warm-on-startup is operator-triggered only; HELM never infers a restart.**
  Pinned does not mean auto-load, so after a server restart everything is cold
  though still marked pinned — fixing that is genuine value. But there is **no
  restart signal** anywhere in the API (no uptime, no start time, no resetting
  counter), and the only available inference — an empty model list — is *also* the
  signature of the router's nuclear option, where a backend failure makes it evict
  everything to clear hardware state. Auto-warming on that inference would have
  HELM mass-loading into a router mid-incident. So: no scheduler, no startup hook,
  no inference. A human presses the button, and even then it confirms first.
- **The NPU is never auto-warmed.** Warming an NPU model evicts whichever NPU
  model is resident (exclusivity), unattended, with nobody present to see the
  warning. NPU models are skipped and *reported*, not silently dropped. Warming is
  single-flight, so two tabs cannot race the router's own eviction logic.
- **The log stream must not die silently.** A long-lived streaming response with
  no bytes flowing is read as a dead origin by an upstream reverse proxy or CDN
  edge and killed — and a quiet log tail is exactly that. HELM emits a keepalive
  comment on an interval and sets the headers that stop intermediaries buffering a
  stream into uselessness. Upstream failures are delivered *inside* the stream, so
  a refused read reads as an error rather than a stream that simply "ended."
- **Log highlighting catches the four events that explain everything else.** In a
  mostly-noise stream, the classifier flags a watchdog action, an ordinary LRU
  eviction, the **mass-eviction nuclear option** (tested *before* ordinary
  eviction, since it also contains the word "unloading" — otherwise a catastrophe
  reads as routine housekeeping), a load taking effect with the backend's own
  ground-truth slot count, and the startup pool figure (with a warning when it
  reads small — the memory saga's exact tell). Classification runs server-side so
  the order-sensitive rules stay the tested ones.
- **Benchmarks run real inference on a shared box, so they are strictly
  operator-triggered** — never on load, never scheduled, never a side effect of
  opening a page. Timing is measured locally rather than read from the global
  stats endpoint, because that endpoint reports the last request from *any* client
  and can report a fabricated rate. Each result stores the model's configuration
  at measurement time, because a rate without its slot/context config is not
  comparable to the next one — and comparing them anyway is how folklore forms.
- **The system panel is detected, never hardcoded.** Processor, GPU family, the
  memory pool (from the same single extractor the dashboard trusts), NPU presence,
  storage, and the installed-vs-installable inference backends are all read live.
  Deploy HELM elsewhere and the panel reports that machine; when the read fails it
  says the box is undetected rather than showing a fabricated one.

---

## 6. Config surface

The config surface is a guarded validate/diff/backup/apply pipeline over two
blast-radius levels. Both run the same sequence: read current, validate, diff,
back up the prior version, confirm with the blast radius named, apply, and keep a
one-click restore point. The validation is pure and table-tested, so the safety
rail is provable in isolation.

### Behavior

- **Level A — per-model recipe options.** Raw JSON editing of one model's options
  (context size, args, backend, pinned). Read-first (the editor opens on the
  running truth, never blank), validate/diff as a dry run, apply with a confirm
  (applying reloads the one model), export, and reset-to-auto.
- **Level B — global server config.** Read and validate/diff the live global
  config. Applying is assumed to require a full server restart that drops every
  model.
- **Backups / restore** — every apply backs up the prior version first; restore
  replays a backup through the same apply pipeline.

### Correctness rules

- **A raw edit is more powerful, not more correct.** The Level A validator runs
  the *same* reserved-argument check the structured form uses, so a raw edit
  cannot smuggle in a flag the form would have blocked. It parses the JSON first
  and refuses outright anything that is not valid — a malformed config is how you
  brick a server that will not restart. Unknown recipe keys are *warned* (Lemonade
  may accept recipe-specific keys HELM has not catalogued), but a typo becomes
  visible rather than being silently ignored by the server. Context-per-slot is
  cross-checked against the model's trained ceiling.
- **Apply reloads, and says so before it does.** A Level A apply reloads the
  model, so it names that blast radius (a several-second drop, in-flight requests
  fail) and requires confirmation. It backs up the prior options first, and reads
  back the result to confirm the change actually took effect.
- **The dangerous-key guard on the global config.** Changing a high-consequence
  global key — the GTT-enable switch, the listen host/port, model directories, or
  anything that looks like a credential — raises a distinct "danger" issue that
  demands an explicit second confirmation. The GTT-enable flag flipping off is
  called out *specifically and by name*: it is the exact switch behind the memory
  saga, making the server see only the small carve-out instead of the full pool. A
  raw editor that let that sail through would be the foot-gun the pipeline exists
  to prevent. Credential-looking keys are flagged without ever echoing their
  values.
- **Reset-to-auto reports honestly when the mechanism is unconfirmed.** There is
  no dedicated reset verb, so reset reloads with saved-options and no overrides and
  then reads back the saved state; if the overrides did not actually clear, it says
  so and flags it, rather than claiming a reset that did not happen.
- **Global apply is deliberately not wired, and refuses loudly.** HELM can read,
  validate, and diff a global change, but it will not apply one until the write
  mechanism is confirmed on the box (an API route to trace, or a file mount it does
  not yet have). Faking an apply here is the exact foot-gun the pipeline exists to
  prevent, so it returns an explicit "not wired" refusal naming the two ways to
  unblock it — and a global *restore* returns the same refusal rather than
  pretending the rail exists where the write does not.
- **Import is an additive merge, never a destructive restore.** Importing operator
  metadata merges rather than deleting anything absent from the payload — treating
  an import as a restore-to-snapshot would silently destroy tags added since the
  export.

---

## Appendix — why this shape

Every rule above traces to the same failure mode: a plausible-looking wrong
number that someone downstream reasoned from. The memory-pool saga (a phantom
ceiling read from the wrong field) is the archetype; the 8x per-slot context lie,
the fabricated throughput rate, the "crashed" claim on a deliberate reload, the
empty tool-badge that looks like "no tools," and the always-on drift indicator
are all the same shape in different corners. HELM's answer is uniform: decide
derived values once in tested code, mark everything unreadable as unknown, keep
the kind of every failure, and never let a write happen without naming what it
will cost. The surfaces differ; the discipline does not.
