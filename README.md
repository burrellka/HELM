# HELM

> A read/write control plane for [Lemonade Server](https://github.com/lemonade-sdk/lemonade). See your local-LLM fleet honestly, manage it safely, and get grounded advice about your models — in one self-hosted web UI.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/u/brain40)
[![Release](https://img.shields.io/badge/release-0.9.7-16a34a.svg)](https://github.com/burrellka/HELM/releases)

HELM sits in front of a Lemonade Server and gives you:

- **A dashboard** that tells the truth about what's loaded — real memory pool, derived per-slot context, throughput, crash detection — and renders "unknown" as *unknown*, never as a confident wrong number.
- **Safe controls** — load, unload, pin, and per-model settings — behind guardrails that refuse honestly (and can run fully read-only if you don't hand it an admin key).
- **A Model Advisor** that explains your models in plain language, *grounded in your actual hardware and inventory* — it teaches concepts freely but never invents a model-specific fact. A fast **Summary**, a deeper **Analyze**, and a streaming **chat** about any model.
- **Operations** — a live log tail, warm-on-startup, benchmarks, and a "current system setup" panel that reads your box live so HELM is portable to any machine by changing one URL.

It runs as a two-container Docker stack. Your models and your data never leave your hardware.

---

## Why HELM exists

Lemonade Server is a fast, capable local-LLM runtime. But operating a fleet of models on it — which is loaded, whether the context you *think* you set is the context that's *actually* running, whether a model will fit, why throughput dropped — meant reading raw JSON or trusting a UI that quietly rounded the hard parts off.

The design principle HELM is built on came out of a real failure mode: **a control plane that displays a confidently wrong number is worse than no UI at all.** If HELM can't be sure of a value, it says so. If a model's slot count can't be parsed, the dashboard shows "unknown" — never a plausible-looking default that happens to be 8× off. Every generated sentence in the Advisor is visibly labeled as generated and traces back to a real, provided fact.

If you run one model and never touch its settings, you may not need HELM. If you run many, tune them, and want to *see what's true* before you act, that's what HELM is for.

---

## How HELM was built (and yes — it was vibe-coded)

HELM was **vibe-coded**: built conversationally, in natural language, by a person directing AI coding agents rather than typing the implementation by hand. It's not a toy that fell out of a single prompt, though — it's the product of a deliberate three-way collaboration, and it's worth being honest about how it came together, because the *method* shaped the result.

**The three of us:**

- **The operator** — owns the homelab, the Lemonade box, and every decision. Set the direction, tested each build against a real fleet, and kept the work honest ("show me it works, don't tell me it works").
- **The architect** ("H.A.L.O.") — owned the spec. Wrote the functional requirements, the correctness rules, and the review that caught the sharp edges *before* they shipped (the parser that could lie by a factor of 8, the memory pool that's easy to read wrong, the capability join that must distinguish "no tools" from "we don't know").
- **The developer** (an AI coding agent) — wrote and tested the code, ran it against the live box, and reported back with evidence.

**The approach that made it work:**

1. **Spec first, then build.** The architect's requirements and review findings (documented internally as F1–F11) became a checklist. Each was implemented *and tested against captured real payloads* before it was called done.
2. **Grounded, never recalled.** The guiding rule for the whole product, and especially the Advisor: state facts only from real, provided sources; teach concepts freely; and when something is unknown, *say unknown*. Small models fabricate under load, so the Advisor's default analyst is a capable model, and its instructions carry an un-editable safety floor.
3. **Verify before claiming.** Nothing was reported "done" on the strength of a green typecheck. Features were exercised end-to-end against the live Lemonade box — the load planner against real context math, the pin route reconstructed from a packet trace, the streaming chat timed for real first-token latency — and the evidence went into the notes.
4. **Fat genius over lean idiot.** When trimming, cut verbosity, never capability or grounding. A cheaper answer that hallucinates is worse than a slower one that's right.

The result is a codebase whose comments read like a lab notebook — they explain *why* a guard exists, often citing the incident that motivated it. That's deliberate. Vibe-coded doesn't have to mean unexamined.

---

## Quick start

You need a running **Lemonade Server** somewhere HELM can reach (same machine or elsewhere on your LAN).

```bash
# 1. Get the compose file and the env template
curl -fsSL https://raw.githubusercontent.com/burrellka/HELM/main/infra/docker-compose.production.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/burrellka/HELM/main/.env.example -o .env

# 2. Edit .env — set HELM_LEMONADE_URL to your Lemonade server.
#    Leave HELM_LEMONADE_ADMIN_API_KEY blank to run read-only (safe first step).

# 3. Up
docker compose up -d
```

Open `http://<this-host>:5080`. The first visit lands on a setup screen — pick an operator password (≥12 chars, no recovery, store it in a password manager). HELM generates its own JWT signing secret on first boot; no key ceremony.

**Read-only first.** With no admin key, HELM shows the whole dashboard and the Advisor, and hides the write controls with a clear note. That's a fully supported mode — a great way to look before you touch. Add `HELM_LEMONADE_ADMIN_API_KEY` to `.env` and restart when you want load/unload/pin/config.

**Portable across machines.** Buying a second box? Deploy the same images there and change one line — `HELM_LEMONADE_URL`. HELM reads that machine's hardware live and describes it correctly; nothing about a specific box is baked into the code.

---

## Documentation

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The technical design: containers, data flow, the shared arg parser, streaming, portability. |
| [docs/FUNCTIONAL-SPEC.md](docs/FUNCTIONAL-SPEC.md) | What each surface does and the correctness rules behind it. |
| [docs/CAPABILITY-GUIDE.md](docs/CAPABILITY-GUIDE.md) | A feature-by-feature walkthrough of the whole UI. |
| [docs/HELP.md](docs/HELP.md) | Task-oriented help — "how do I…" for common operations. |
| [infra/npm-mtls-ingress.md](infra/npm-mtls-ingress.md) | Optional: putting HELM behind Nginx Proxy Manager + Cloudflare mTLS, with the streaming config that trips everyone up. |

---

## The stack

- **Backend** — FastAPI (Python 3.12), `httpx` to Lemonade, SQLite for HELM's own state (operator metadata, settings, saved summaries). The container reports its own version, so "which image is running?" is always answerable.
- **Frontend** — React + Vite + TypeScript, dependency-light on purpose (its own compact, safe Markdown renderer rather than a heavyweight library). Served by nginx, which also proxies `/api`.
- **Two containers** — `helm-web` (nginx + SPA) and `helm-api` (uvicorn). One bind-mounted `/data` volume holds everything HELM can't rebuild by re-reading the box — so back it up.

Images are published to Docker Hub as `brain40/helm-api` and `brain40/helm-web`. Tags are immutable; `latest` always equals the newest version tag byte-for-byte.

---

## Security posture

- **Its own operator login.** Even behind a reverse proxy, HELM authenticates operators itself, because a control plane that can unload a model other services depend on — or delete gigabytes of weights — should never be one misconfigured proxy away from the open internet.
- **The admin key stays server-side.** It gates writes and is never sent to the browser. Read-only mode is a first-class supported state, not a degraded one.
- **No secrets in this repo.** Keys live in your `.env` (gitignored). The sample compose and env template ship with placeholders only.

---

## Status

HELM is actively developed against a live fleet. It has a read-only dashboard, full control surface, library management, operations tooling, and the Model Advisor with streaming chat. See the [release badge](https://github.com/burrellka/HELM/releases) for the current version.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE). Use it, fork it, run it on your own hardware. No warranty; it manages a real inference server, so read what a control does before you click it.

---

*Built by a human operator, an AI architect, and an AI developer, against a real homelab. If you're one of the many people running local models and wishing you could just **see** what your server is doing — this was built for you too.*
