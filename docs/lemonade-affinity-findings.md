# Lemonade session / slot affinity — findings for HALO + KAIROS

**Date:** 2026-07-25 · **By:** the HELM crew · **For:** the architect (HALO) and the KAIROS crew

**The question KAIROS asked:** can we pin a conversation to a warm slot so we stop
eating cold ~14K-token prefills every turn? Which layer, if any, gives us that?

---

## Bottom line

**There is no "session pinning" feature to turn on — it doesn't exist as a concept.**
llama.cpp (and therefore Lemonade) has no session or conversation id. All affinity is
**slot-based**. So "pin a session" means one of exactly three things, and here is where
each stands through Lemonade **today**:

| Mechanism | In llama.cpp? | Through Lemonade now? | To enable |
|---|---|---|---|
| **Automatic prefix cache** (LCP slot match, `--slot-prompt-similarity`, default 0.10) | Yes | **Yes — works today** | Nothing. Keep the prompt prefix byte-stable. |
| **Durable KV save/restore** (`/slots/{id}?action=save\|restore`) | Yes | Endpoint **proxied but disabled** | `--slot-save-path` on the box — **HALO's lane** |
| **Deterministic slot pin** (`id_slot` request param) | Yes | **No — Lemonade strips it** | Upstream Lemonade change to forward `id_slot` — **an ask to lemonade-sdk, not a box toggle** |
| **Session / conversation-id affinity** | **No — does not exist** | n/a | Would have to be built in the orchestration layer |

**Near term:** KAIROS needs nothing from HALO. The automatic prefix cache already reuses
the full prefix warm (measured **896 ms → 66 ms, 13×**, sustained). The fix is prefix
discipline (below). **Durable/deterministic** pinning needs two box/upstream changes that
are **complementary — both required**, and KAIROS has neither today.

---

## Evidence

All tests: one loaded llamacpp model (Gemma-4-26B, 8 slots), reads/completions open (no key).
The actionable per-request signal is `usage.prompt_tokens_details.cached_tokens` and the
`timings` block (`cache_n`, `prompt_n`, `prompt_ms`) — how many prefix tokens hit warm cache.

### A. Does a response hand back a session/slot handle? — No.
Body carries only the standard per-request `id` (`chatcmpl-…`, fresh each call, not reusable)
and a constant `system_fingerprint` (build hash). No `slot_id`/`session`/cache id in body or
headers. But `cached_tokens`/`timings` give you cache-hit observability per request.

### B. Automatic prefix cache — works, and prefix stability is the lever.
- 600-token identical prompt, repeated: req1 cold (`cached_tokens=7`, `prompt_ms≈897`) →
  req2+ warm (`cached_tokens=633`, `prompt_ms≈66`). Sustained across many requests.
- Works **with no special params** (no `id_slot`, no `cache_prompt` — `cache_prompt`
  defaults to true).
- **Decisive:** changing just the **first token** dropped `cached_tokens` 270 → 7 and
  re-prefilled the whole prompt. A single volatile token at the front throws away the
  entire 14K prefill. This is almost certainly why KAIROS eats cold prefills today.

### C. Durable KV save/restore — exists and is proxied, disabled on the box.
Lemonade proxies llama.cpp's slot endpoints (all four prefixes `/api/v0`, `/api/v1`,
`/v0`, `/v1`):
- `GET /v1/slots` — state
- `POST /v1/slots/{id}?action=save` — persist a slot's KV to a file
- `POST /v1/slots/{id}?action=restore` — restore it
- `POST /v1/slots/{id}?action=erase`

`POST /api/v1/slots/0?action=save` returns a **real passthrough** error:
> `501 — "This server does not support slots action. Start it with --slot-save-path"`

So it's live, just not enabled. **Enable:** `lemonade config set
llamacpp.args="--slot-save-path /var/lib/lemonade/slot-saves"` + reload.

> ⚠ Earlier these looked like they didn't exist because they were probed with **GET** — they
> are **POST**, and a GET falls through Lemonade's extensionless catch-all to the SPA
> `index.html` (a false 200). Always POST, and check the body, not the status.

### The `id_slot` pin — Lemonade strips it. (Three independent proofs.)
llama.cpp honors `id_slot` ("assign the task to a specific slot; -1 = idle; default -1").
But it never reaches the backend through Lemonade:
1. **A→slot0, B→slot0, A→slot0**: A came back warm — impossible if both pinned to slot 0
   (B would have evicted A). So the pin wasn't applied.
2. Caching is **identical with and without** `id_slot`.
3. **`id_slot:99`** (out of range — only 0–7 exist) returns a **clean 200 completion**, not
   llama.cpp's invalid-slot error. If Lemonade forwarded it, the backend would reject 99.

### No higher-level session concept.
llama.cpp server docs: *"There is no higher-level session concept — slot affinity is
entirely slot-based."* No `/sessions` route on Lemonade either
(`/v1/sessions`, `/api/v1/sessions` → 404). `/props` is **not** proxied
(`/props` → catch-all SPA, `/api/v1/props` → 404), so KV-cache type isn't reachable either.

---

## Recommendations

### 1. KAIROS, now (no dependencies): make the prompt prefix byte-stable.
The automatic cache keys on the leading tokens. To keep the 14K prefill warm turn-to-turn:
- Serialize the system prompt + prior turns **identically** every turn (append-only). No
  timestamps, no regenerated preamble, no re-ordered context **early** in the prompt — put
  anything volatile at the **end**.
- **Verify** with `cached_tokens`: turn 2+ of a session should show `cached_tokens ≈ prefix
  length`. If it's ~7, something broke the prefix that turn.
- Keep concurrent **distinct-prefix** sessions **≤ slot count (8)**. Each slot holds one
  session's cache; beyond 8 they evict each other and cold prefills return.

This alone should kill most of the cold-prefill pain.

### 2. HALO, box-side: enable `--slot-save-path`.
Unlocks durable save/restore so a session's KV survives eviction and model reload (it's a
file). `lemonade config set llamacpp.args="--slot-save-path …"` + reload.

### 3. Architect, upstream: raise `id_slot` passthrough with lemonade-sdk.
Lemonade strips `id_slot`, so completions can't be routed to a chosen slot. Without it,
save/restore is only **half** a mechanism — you can restore a session to slot N but can't
guarantee the next completion runs on slot N (Lemonade's router picks by LCP/idle). For
robust durable pinning you need **both** `--slot-save-path` **and** `id_slot` forwarding.

### Open test once #2 lands (before betting on save/restore for pinning)
Does `restore slot N → fire a completion` actually reuse slot N's restored cache, given
`id_slot` is stripped? If the router lands the completion on a different slot, restore buys
nothing for pinning. Test: restore a known prompt to slot 0, fire that exact prompt, check
`cached_tokens` is full. This is the hinge and it's currently untestable (feature disabled).

---

## Sources
- llama.cpp server README — `id_slot`, `cache_prompt`, `--slot-prompt-similarity`, slot
  save/restore, "no higher-level session concept":
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Lemonade llama.cpp-specific API (slot endpoints, prefixes, `--slot-save-path`):
  https://lemonade-server.ai/docs/api/llamacpp/
- llama.cpp persistent-KV-per-session tutorial:
  https://github.com/ggml-org/llama.cpp/discussions/20572
- Lemonade issue — support llama.cpp `/slots` endpoints:
  https://github.com/lemonade-sdk/lemonade/issues/1715
