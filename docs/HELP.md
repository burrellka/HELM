# HELM — How-To & Troubleshooting

Practical answers for running HELM, the self-hosted web UI that sits in front of
a Lemonade Server and lets you see and manage your local-LLM fleet honestly.

This is a task-oriented FAQ. If you want the design rationale instead, see
`docs/ARCHITECTURE.md` and `docs/FUNCTIONAL-SPEC.md`. Throughout, `<helm-host>`
means whatever machine you run HELM on, and `<lemonade-host>` means wherever your
Lemonade Server listens.

---

## Setup & access

### How do I install HELM?

Grab the production compose file and the env template, edit one line, and bring
it up:

```bash
curl -fsSL <repo>/infra/docker-compose.production.yml -o docker-compose.yml
curl -fsSL <repo>/.env.example -o .env
# edit .env: set HELM_LEMONADE_URL to your Lemonade server
docker compose up -d
```

Then open `http://<helm-host>:5080`. The first visit lands on a setup screen
where you choose an operator password. HELM generates its own session-signing
secret on first boot, so there is no key ceremony. You need a reachable Lemonade
Server before HELM is useful, but HELM will start and show its read-only surface
even if Lemonade is down.

### Do I need an admin key? What is the difference between read-only and management mode?

No — HELM runs fine without one, and that is a supported mode, not a degraded
one. With no `HELM_LEMONADE_ADMIN_API_KEY` set, HELM shows the full dashboard and
the Model Advisor, hides the write controls, and tells you why. This is the safe
way to look before you touch.

Set `HELM_LEMONADE_ADMIN_API_KEY` in `.env` and restart to unlock **management
mode**: load, unload, pin, per-model settings, and warm. The key is the same one
that gates Lemonade's own management routes. HELM's backend holds it and never
sends it to the browser. Read-only vs management is decided entirely by whether
that key is present.

### I forgot my operator password — how do I reset it?

There is no password recovery, by design — this is a single-operator homelab
tool. To reset, delete the auth file and re-run setup:

1. Stop HELM (`docker compose down`, or stop the `helm-api` container).
2. In the host directory you bind-mounted to `/data`, delete `auth.json`.
3. Start HELM again. The next visit shows the first-boot setup screen and you
   pick a new password.

`auth.json` is deliberately a small JSON file (one password hash and a
setup flag) rather than a row in the database, precisely so you can clear it
with a file delete at midnight without needing a database client. Deleting it
does not touch your models, benchmarks, tags, or advisor settings — those live
in the database in the same directory. Store the new password in a password
manager; length is what matters (minimum 12 characters), not symbol soup.

### Why does the login cookie not stick, or say the connection is "non-secure"?

Almost always because your reverse proxy is not forwarding the
`X-Forwarded-Proto` header. HELM terminates behind a proxy that speaks plain
HTTP to the container, so it has to be *told* the browser is on HTTPS before it
will mark the session cookie `Secure`. If it cannot tell, the login screen shows
a warning that the connection "is not detected as HTTPS."

- On `localhost` over plain HTTP this warning is expected and harmless.
- Behind a proxy it means the proxy host is not passing
  `proxy_set_header X-Forwarded-Proto $scheme;`. Add it. HELM also falls back to
  the `Origin` and `Referer` headers on the actual login POST, so login usually
  still works — but the cookie rides the internal leg non-`Secure` until you fix
  the header, so fix it.

### How do I point HELM at a Lemonade server on a different machine?

Change one line: `HELM_LEMONADE_URL` in `.env`, then restart. Examples:

```
HELM_LEMONADE_URL=http://localhost:13305        # Lemonade on the same box
HELM_LEMONADE_URL=http://<lemonade-host>:13305  # Lemonade elsewhere on the LAN
```

If Lemonade runs as a host service rather than a container, use the host's LAN
address rather than a container name. HELM reads that machine's hardware live and
describes it correctly, so the same images are portable to any box by changing
this URL alone — nothing about a specific machine is baked into the code.

---

## Daily use

### How do I load, unload, or pin a model?

All three live on the **Dashboard**, on each model's row (and load is available
for downloaded-but-cold models). They require management mode (an admin key).

- **Load** opens a dialog that previews exactly what will happen — including any
  model that will be evicted — before you commit.
- **Unload** warns you first if the model is pinned or has requests in flight,
  and reminds you a reload costs roughly 10–20 seconds and does not come back on
  its own after a Lemonade restart.
- **Pin / unpin** is a metadata flag now, not a reload: toggling it no longer
  drops the model. HELM re-reads health afterward and confirms the flag actually
  flipped rather than trusting the call — if it cannot confirm, it says so
  instead of reporting a false success. Pinning means "do not evict"; it does
  **not** mean "auto-load after a restart" (see warming).

### How do I change a model's context, slots, or sampling settings?

Open the model's settings/load dialog. You set values in the units you think in,
and HELM does the math:

- **Slots (concurrent requests)** and **Context per slot** are separate fields.
  You never multiply them yourself — HELM computes the total `ctx_size` Lemonade
  actually wants and shows it back to you as a read-only consequence ("8 slots ×
  64K → 524288"). That multiplication is the number that misled people for
  months; the dialog exists so you never do it by hand.
- **Sampling** (temperature, top-k, top-p, min-p, repeat-penalty) each has its
  own labeled field. You never type raw `--flags`; HELM composes them. Setting
  the same value in both a field and the raw-args escape hatch is refused,
  because two sources disagree silently depending on arg order.
- Slots are a llama.cpp concept. On NPU recipes the slot controls are hidden
  (not greyed out) because the setting genuinely does not exist there.

Applying any change reloads the model (~10–20s), because Lemonade has no
change-in-place call. HELM names that blast radius and asks you to confirm, backs
up the prior options first, and reads back the result to confirm it took.

### Why did my save-as-default not persist, or why does the dashboard show "drift"?

Because running config and saved config are two different things, and HELM shows
you both. A model can run with options that were never persisted — e.g.
`lemonade load X --parallel 8` runs with 8 slots but saves nothing, and a
Lemonade restart silently reverts to whatever was saved (possibly nothing).

The dashboard flags this as two distinct classes:

- **Not saved** — the model is running with something that a restart will drop.
  Quieter, common (Lemonade auto-detects the backend, so that field is always
  "not saved" and is marked benign).
- **Drift** — saved and running both set a value and they *disagree*. The louder
  one; the running model is not what its saved config says.

If you wanted a setting to survive restarts, use the save-options / persist path
in the settings dialog rather than a bare load. If drift persists after you
saved, the read-back will tell you whether Lemonade actually accepted the change.

### What do "unknown" and "unavailable" mean on the dashboard?

They mean exactly that, and they are deliberate. HELM's core rule is that a
confidently wrong number is worse than no number. So:

- If a model's slot count cannot be parsed, the cell reads **unknown** rather
  than a plausible-looking default that might be 8× off.
- The NPU utilization gauge is a known-broken upstream reading on some hardware,
  so it renders **unavailable** rather than a fake 0%.
- The memory "source" line shows the exact field a number came from, so you can
  trace it instead of trusting it.

When HELM can't be sure, it says so on purpose. That is the feature, not a bug.

### How do I use the Model Advisor? When do I use Summary vs Deep analysis vs the chat?

The Advisor explains your models in plain language, grounded in your real
hardware and inventory — it teaches concepts freely but never invents a
model-specific fact. Every generated sentence is labeled as generated and traces
back to a provided fact. Three surfaces, escalating in cost:

- **Summary** — a fast, 2–4 sentence read of one model built from its facts card
  only (no inventory, no web search). Use it for a quick "what is this." A small,
  fast model runs it; target is a couple of seconds. Summaries can be saved so
  they show instantly next time.
- **Deep analysis (Analyze / synthesize)** — a thorough, beginner-friendly
  writeup that also weighs your *other* models as alternatives and checks fit
  against your hardware. Uses the big analyst model. Slower.
- **Chat** — ask follow-up questions about a specific model, anchored to its
  facts. Streams tokens as they arrive.

Layer 1 facts (params, quant, size, fit) are deterministic and render instantly
with no model running. Only the generated layers cost time.

### The advisor chat is slow — how do I make it faster?

Some of this is inherent, some is tunable:

- **Summary** is already fast (a small non-reasoning model). Reach for it when
  you don't need the deep writeup.
- **Deep analysis and chat** use the big analyst and stream, so the first words
  appear in a second or two even though the full answer can take 1–2 minutes.
- The **first question in a session pays a one-time cost** — the model's cold
  first token on a large context lands around 14 seconds.
- In **Advisor settings** you can pick a smaller or non-reasoning analyst. A
  reasoning model spends part of its budget "thinking" before it writes; a
  non-reasoning 8B answers sooner.

### How do I enable web search in the advisor?

Set `HELM_WEB_SEARCH_KEY` in `.env` to a Google Gemini API key and restart. It is
a separate credential from any model or admin key, with its own quota (the free
tier is generous). With it set, the advisor runs a real grounded web lookup and
cites sources; without it, the advisor runs fine and says plainly that no live
lookup was done rather than inventing one. You can verify the key end-to-end with
the "test" action in Advisor settings, which runs a real search and returns a
sample (never the key). The provider and model are editable in the UI and stored
in the database, so you can swap to a newer grounding model without a redeploy.

### How do I edit the advisor's instructions or knowledge file?

In **Advisor settings**. The **instructions** ("persona") and the **knowledge
file** are both editable text boxes; send an empty value (or the restore action)
to fall back to the shipped defaults. You can also upload a knowledge file
directly (UTF-8 text, kept under 200,000 characters so it fits the context
budget). Note that the safety kernel — the "do not fabricate, say unknown" floor
— is a code constant and cannot be edited away; no instruction change can turn it
off.

---

## Troubleshooting

### "Management disabled" / writes refuse — why?

HELM has no Lemonade admin key, so it is in read-only mode. Every write path
refuses locally with a clear message rather than firing a request that would come
back as a bare 401. Set `HELM_LEMONADE_ADMIN_API_KEY` in the stack's `.env` and
restart. If you *did* set it and writes still refuse, check that the value
matches the key Lemonade actually expects and that the container picked up the
new `.env` (a plain `up -d` re-reads it; a bare restart may not).

### The app won't start with a SQLite / "unable to open database" error.

This is almost always a permissions mismatch on the bind mount. HELM writes its
database, session secret, and auth file to `/data`, and it runs as a non-root
user. If that user cannot write the host directory you mounted at `/data`, SQLite
can't open the database and the app refuses to boot.

Fix it by matching the container's user to the directory's owner. Check the owner
on the host (`stat -c '%u %g' <your-helm-data-dir>`), then set `PUID`/`PGID` in
`.env` to those numbers and restart. Note these feed Docker's own `user:`
directive — this is **not** the LinuxServer.io convention, so setting them as
plain environment variables alone would do nothing; the compose file wires them
into `user:` for you. The simplest alternative is to `chown` the data directory
to the uid/gid you want HELM to run as.

### The live log stream shows nothing, or dies after ~100s behind a proxy.

Streaming fails in two proxy-specific ways, and both look like an application
bug when they aren't:

1. **Buffering** — your proxy's nginx holds the whole response and flushes at the
   end, so the log pane shows nothing while the backend looks perfect. Fix:
   `proxy_buffering off` on the streaming location.
2. **Idle timeout** — an edge (e.g. Cloudflare) kills a response after ~100s of
   silence, and a quiet log tail at 3am is exactly that. HELM emits a keepalive
   byte every ~15s to defeat this, but it only helps if the proxy isn't buffering
   it away, and you also need a long `proxy_read_timeout`.

HELM's own container nginx already does the right thing; a proxy *in front of it*
needs the same treatment, and there is no checkbox for it. The exact location
blocks to paste are in `infra/npm-mtls-ingress.md`. The test that matters: open
the stream on a quiet log and leave it for more than two minutes — if it dies
around 100s, the keepalive isn't reaching the edge.

### A model won't fit, or the advisor says it "will not fit" — what now?

HELM computes fit from the model's footprint against your memory pool and reports
one of three verdicts:

- **Fits alongside what's loaded** — enough is free right now.
- **Fits only if something is evicted** — bigger than the current free space but
  smaller than the whole pool. Unload or let the router evict an unpinned model
  first, then load.
- **Too big for the pool** — larger than the entire pool; it will not load here.
  Your options are a smaller or more-heavily-quantized variant, freeing the pool
  by unloading others, or different hardware.

Remember context adds on top of the weights, and HELM deliberately does not
invent a precise KV-cache number. For NPU models that carry no catalog size, fit
is "unknown" until you load them once and HELM can measure the real footprint.

### The advisor spent its budget "reasoning" and produced no answer.

Reasoning models emit a hidden reasoning channel *before* the actual answer. On a
tight token budget the reasoning can eat the whole allowance, so the answer comes
back empty. HELM detects this exact case and tells you, rather than showing a
blank. The fix is to pick a **non-reasoning analyst** (an 8B is a good choice) in
Advisor settings — or just ask again. The Summary model is non-reasoning by
default for this reason. Your local facts card is unaffected either way.

### The dashboard shows an absurd throughput like 1,000,000 tok/s.

That is an artifact, not a measurement — it's what you get dividing a tiny token
count by a near-zero elapsed time after a very short response. HELM shows the
reported number but flags it as implausible and tells you to treat it as no
measurement rather than a real rate. For a comparable figure, run a **benchmark**
from the Operations page: it times inference itself (rather than reading a global
counter any client can have moved) and stores the model's configuration alongside
each result so numbers stay comparable over time.

### Why won't HELM auto-load my pinned models after a restart?

Because it cannot safely tell a restart apart from an incident. Lemonade exposes
no restart signal, and an empty model list — the only thing HELM could infer from
— is *also* what the router shows mid-incident while it deliberately clears
hardware after a backend failure. Auto-loading into that would make things worse.

So warming is operator-triggered only: add models to an explicit **warm set** on
the Operations page and press the button when you know the box is idle. NPU
models are never warmed unattended (warming one would evict the resident NPU
model with nobody watching). Pinned means "do not evict," which is a different
intent from "bring this back after a restart."

### How do I back up HELM's data?

Back up the host directory you mounted at `/data`. It holds HELM's database
(operator descriptions, tags, favorites, benchmark history, advisor settings and
saved summaries), the session-signing secret, and the operator password hash.
Operator metadata has **no upstream source of truth** — Lemonade does not know
about it — so if you lose this directory, it is gone. A file copy while the
container is stopped is the safest snapshot; the database uses WAL mode, so if
you copy it live, grab the `-wal` and `-shm` sidecar files too.

### How do I upgrade HELM?

Pull the new images and recreate:

```bash
docker compose pull
docker compose up -d
```

Image tags are immutable (pinned to a version), so an upgrade is a deliberate
tag bump in the compose file, not a silent change under a `latest` tag. Your
`/data` directory carries across untouched, so settings, tags, and benchmark
history survive the upgrade. If you pin versions in the compose file, bump the
tag there before pulling.
