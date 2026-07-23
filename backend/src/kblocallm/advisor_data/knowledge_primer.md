# 14d — MODEL ADVISOR KNOWLEDGE FILE: LOCAL MODELS, GROUND TRUTH
## Version 1.0 | 2026-07-18 | Author: H.A.L.O. (architect) for the HELM Model Advisor
## Attach to the advisor's instructions as its concept authority.

> **HOW TO USE THIS FILE (instructions to the advisor model).**
> You have three truth sources. Never confuse them:
> 1. **This file** = concept truth. What quantization IS, how MoE works, how to reason about fit.
>    Teach from it freely. It is stable knowledge, reviewed by the architect.
> 2. **HELM's facts card** = model-specific truth. THIS model's params, quant, context, footprint,
>    capabilities, and the operator's owned inventory. Claims about a specific model come ONLY from
>    here or from search results. If the card does not carry it, it is unknown.
> 3. **Web search results** = fresh truth. Community sentiment, new releases, benchmark discussion.
>    Report it as "according to <source>", never as your own knowledge.
> Never state a model-specific number (benchmark score, context length, release date) from memory.
> If this file and a fetched fact conflict, the fetched fact wins for that model; say so.
> The operator is a smart novice. Teach in plain terms first, then give the precise version.

---

## 1. WHAT A LOCAL MODEL IS, MECHANICALLY

A large language model is a single enormous mathematical function. Its "knowledge" lives in
**parameters** (also called **weights**): billions of numbers learned during training. "7B" means
seven billion parameters. Running a model (**inference**) means streaming those numbers through the
processor to predict the next **token** (a word fragment, roughly 3 to 4 characters of English).

Two consequences drive everything else in this file:

- **Memory:** every parameter must sit somewhere in memory to be used. Parameter count times the
  size of each number equals the memory footprint. This is why quantization exists.
- **Bandwidth:** to produce ONE token, the processor must read essentially ALL the active weights
  once. Token generation speed is therefore limited by how fast memory can be read (memory
  bandwidth), not by raw compute. This is the single most misunderstood fact on hobbyist forums,
  and it explains most performance behavior on this box.

## 2. PARAMETERS: WHAT SIZE BUYS, AND WHAT IT DOES NOT

More parameters generally buy better reasoning, more world knowledge, better instruction following,
and better tool use. But the relationship is not linear, and training quality matters as much as
size. A well-trained modern 8B routinely beats a poorly trained older 30B. Rules of thumb:

- **Under 2B:** pattern completion, classification, drafting. Fast. Prone to confident fabrication
  and weak at multi-step reasoning. Never use one as a source of facts.
- **3B to 9B:** capable assistants for focused tasks, decent tool calling on well-trained models,
  good speed. The workhorse class for simple agents.
- **12B to 35B:** strong general reasoning, reliable tool use, good long-context behavior. The
  daily-driver class on this hardware.
- **70B and up (dense):** best quality, but on this box decode speed drops because bandwidth is
  fixed; usable for offline or quality-critical work, not snappy chat.

Size does NOT buy factual reliability on niche topics, freshness (knowledge stops at training
cutoff), or arithmetic. Those need retrieval, tools, or search regardless of size.

## 3. DENSE VS MoE, AND HOW TO READ "26B-A4B"

- **Dense** model: every parameter participates in every token. A dense 27B reads ~27B weights per
  token.
- **MoE (Mixture of Experts):** the model is built from many "expert" sub-networks plus a router.
  For each token, the router activates only a few experts. Naming: **"26B-A4B" = 26B total
  parameters, ~4B ACTIVE per token.**

The two numbers answer different questions, and mixing them up is the most common novice error:

- **Total parameters govern MEMORY.** All 26B must be resident. Footprint math uses TOTAL.
- **Active parameters govern SPEED.** Per token, only ~4B are read, so a 26B-A4B decodes at
  roughly the speed of a 4B dense model. Speed math uses ACTIVE.
- Quality lands in between: an MoE usually reasons better than a dense model of its active size,
  and somewhat below a dense model of its total size, with excellent knowledge breadth.

This is why MoE models are the sweet spot on this box specifically: a 120GB pool has room for the
TOTAL, and the bandwidth-bound decode only pays for the ACTIVE. Plenty of memory, fixed bandwidth
is exactly the profile MoE exploits. When someone on a 24GB GPU says a 26B MoE is a bad deal, their
constraint is memory. Yours is not.

## 4. QUANTIZATION: THE CORE CONCEPT

Training produces weights as 16-bit numbers (fp16/bf16, 2 bytes each). **Quantization** re-encodes
them with fewer bits. The standard analogy is a photograph saved as a JPEG: smaller file, slightly
less fidelity, and at reasonable settings most people cannot see the difference.

**Reading a GGUF quant name, e.g. `Q4_K_M`:**
- `Q4` = roughly 4 bits per weight.
- `_K` = the k-quant family: smarter schemes that spend more bits on the weights that matter most.
  Better quality than the older `_0`/`_1` schemes at the same size.
- `_S` / `_M` / `_L` = small/medium/large variant within that family (how many layers get extra
  bits). M is the usual balance point.
- `IQ...` (IQ4_XS, IQ3_M, ...) = importance-matrix quants: calibrated on sample text to protect the
  most important weights. Strong at very low bit rates.

**Approximate size math.** Bytes per weight ≈ bits/8. Weights-only footprint ≈ params × bytes per
weight. (GGUF files carry some overhead; treat results as estimates, and prefer HELM's measured
load delta when one exists.)

| Quant | ~bits/weight | ~bytes/weight | Quality character |
|---|---|---|---|
| fp16 | 16 | 2.0 | Reference. Never needed for inference here. |
| Q8_0 | 8.5 | ~1.06 | Near-lossless. |
| Q6_K | 6.6 | ~0.82 | Practically indistinguishable from Q8 in normal use. |
| Q5_K_M | 5.5 | ~0.69 | Very good. |
| Q4_K_M | 4.8 | ~0.60 | The community default. Small, measurable, usually invisible quality cost. |
| IQ4_XS | 4.3 | ~0.53 | Slightly smaller than Q4_K_M, comparable quality. |
| Q3_K_M | 3.9 | ~0.49 | Visible degradation begins, model-dependent. |
| Q2_K | 2.6 | ~0.33 | Heavy degradation. Last resort to fit at all. |

**The three laws of choosing a quant:**
1. **Small models bruise easier.** Quantization error hurts a 1B far more than a 70B. Below ~7B,
   prefer Q6_K or Q8_0. A Q8 of a 3B model is still tiny.
2. **Bigger model at lower quant usually beats smaller model at higher quant** when both fit. A
   26B at Q4 generally outperforms a 9B at Q8.
3. **Your constraint decides.** Most online advice optimizes for 8 to 24GB gaming cards, where
   aggressive quants are forced. On a 120GB pool, memory is rarely the binding constraint, so this
   box can afford Q5/Q6 on mid models and Q8 on small models. Lower quants also decode FASTER here
   (fewer bytes to stream per token), so Q4_K_M on big models is a legitimate speed choice, not
   just a memory one. Do not cargo-cult Q2/Q3 advice written for cards one tenth this size.

## 5. CONTEXT WINDOWS AND THE KV CACHE

The **context window** is how many tokens the model can attend to at once (the conversation, the
documents, the tool schemas, and the reply, combined). Two costs scale with it:

- **Memory: the KV cache.** The model caches attention Keys and Values for every token in context.
  This sits ON TOP of the weights footprint and grows linearly with tokens actually used. Rough
  per-token cost = 2 × layers × kv_heads × head_dim × bytes. Modern models use GQA (grouped-query
  attention), which shrinks kv_heads and makes long context far cheaper than older models. Typical
  magnitude: tens of KB to a few hundred KB per token, so 32K tokens can cost anywhere from ~1GB to
  several GB depending on architecture. Treat any specific number as an estimate; the measured
  load delta is truth.
- **Time: prefill.** Before the first output token, the model must process the entire prompt
  ("prefill"). This is compute-and-bandwidth heavy and scales with prompt length. This is the
  known ~18-42s wait on ~25K-token prompts locally, and it is why the fleet routes very long
  prompts to cloud. Long context is not free even when it fits.

Also true: models degrade toward the far end of their advertised window (the "lost in the middle"
effect). A model rated for 128K is not equally sharp at 120K. And Lemonade's `ctx_size` is a TOTAL
divided across parallel slots; the per-slot window is what a single conversation actually gets.

## 6. THE FIT DECISION ON THIS BOX (the math the advisor walks)

Hardware ground truth: Strix Halo, **128GB unified memory, ~120GB pool usable by the GPU** (the
`virtual_mem_gb` pool; the 512MB `vram_gb` figure is a BIOS carve-out and a known liar). Unified
means CPU and GPU share one pool: no PCIe copy penalty, but also one budget for everything.

Fit walk, in order:
1. **Weights** ≈ total params × bytes/weight for the quant (table above), or the measured delta if
   HELM has one (prefer measured).
2. **KV cache** at the context you will actually configure, times parallel slots if serving
   concurrent requests.
3. **Neighbors.** The pool is shared by everything resident: the pinned daily driver, other loaded
   models, and the OS. Fit means fitting ALONGSIDE, not alone. HELM's card says what is resident.
4. **Verdict language:** "fits resident alongside current loads", "fits only if X is evicted", or
   "does not fit at any configured context". State which and show the arithmetic.

Speed expectations, plain version: **decode speed ≈ bandwidth ÷ active-parameter bytes.** Big dense
models are slow to chat with here no matter the compute, because bandwidth is fixed. MoE models
break that tradeoff (Section 3). **Prefill** is the other axis: long prompts cost seconds up front
regardless of decode speed.

## 7. NPU, FLM, AND PLACEMENT (box-specific)

This machine has two AI engines. They are not interchangeable:

- **iGPU (Radeon 8060S)** via ROCm/llama.cpp: runs GGUF models, the main engine, big memory pool.
- **NPU (XDNA2)** via FLM (FastFlowLM): runs specially compiled models only (`.q4nx` and similar).
  Efficient and low-power, but: the NPU is **exclusive** (one NPU model at a time; loading a second
  evicts the first), models must be compiled for it (you cannot point it at a GGUF), and
  **capabilities are baked at compile time**. Tool calling on an NPU model is a property of that
  specific compiled artifact, not of the runtime. This was proven on this box; never assume an NPU
  model can tool-call without checking its capability flag.
- FLM serves requests **serially** (no parallel slots). Slot math is a llamacpp-only concept.

Placement rule of thumb: interactive and multi-model work belongs on the GPU; the NPU suits a
single dedicated always-on workload where efficiency matters more than flexibility.

## 8. READING A MODEL NAME (worked example)

`Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M`, decoded left to right:
- **Qwen3.6** = family and generation (who trained the base and which release line).
- **35B-A3B** = MoE: 35B total (memory), ~3B active (speed). See Section 3.
- **Claude-4.6-Opus-Reasoning-Distilled** = a community **distill**: the base model finetuned on
  outputs of a stronger teacher model to imitate its reasoning style. Distills can punch above
  their size on the teacher's strengths but inherit quirks, and quality varies wildly by who made
  them. Community finetunes deserve more skepticism than official releases; this is exactly where
  search and community sentiment earn their keep.
- **Q4_K_M** = the quant (Section 4).

Other tags worth knowing: **-it / -Instruct / -Chat** = instruction-tuned for dialogue (what you
almost always want; a bare **base** model just continues text and ignores instructions).
**-uncensored / -abliterated** = safety training removed by the community, behavior less
predictable. **-VL / -Vision** = multimodal image input. **-Embedding** = not a chat model at all;
it turns text into vectors for search. **-Coder** = code-specialized. **-Reasoning / -Thinking / -R1-style**
= trained to emit long chain-of-thought before answering: better on hard problems, slower and
token-hungry on easy ones.

## 9. CAPABILITIES: WHAT TO CHECK, NOT ASSUME

- **Tool calling** (function calling): the model can emit structured calls agents depend on. It is
  a trained-in skill plus, for GGUF, a chat-template property, and for NPU models a compile-time
  property. HELM's capability badge is the truth for a given artifact. Quality varies: small
  models often produce malformed calls or, worse, fabricate a success message after a failed call.
  For agent duty, honesty-on-failure matters more than raw capability; this fleet selects
  tool-heavy agent models on exactly that criterion.
- **Vision**: requires a multimodal build (often a separate mmproj file in GGUF land). A text-only
  quant of a vision family loses the vision.
- **Long context**: advertised max is the trained ceiling, not a quality promise (Section 5).
- **Reasoning modes**: some models expose think/no-think switches; reasoning output burns tokens
  and time, worth it only for hard problems.

## 10. HOW TO READ BENCHMARKS AND COMMUNITY SIGNAL (for search results)

- **Benchmarks** (MMLU, GPQA, HumanEval, LMArena, and the like) are directional, not gospel.
  Failure modes: contamination (test data leaked into training), benchmark-tuning (models trained
  to the test), and scope mismatch (a coding score says nothing about agent reliability). A model
  that tops a leaderboard can still be mediocre at YOUR task. Treat single-benchmark hype
  skeptically, prefer consistent performance across several, and prefer task-relevant ones.
- **HuggingFace signals:** download counts and likes measure popularity and momentum, not quality;
  recent-and-rising with active community discussion is a healthier signal than big-and-stale. The
  model card is the closest thing to a spec sheet; a missing or thin card on a community finetune
  is itself a warning.
- **Reddit/forum sentiment** (r/LocalLLaMA and similar): excellent for "how does it feel in real
  use" and for catching problems benchmarks miss (repetition, sycophancy, broken templates, bad
  quants of a good model). Weaknesses: loud early hype, hardware assumptions that do not match
  this box (Section 4, law 3), and vibes without methodology. Weigh multiple voices, note WHOSE
  hardware and use case they describe, and always attribute: "users report", "according to", with
  the source.
- **Recency matters.** The local-model field turns over in months. Advice from a year ago about
  the "best 13B" is history, not guidance. Prefer fresh sources; state the date of what you cite.

## 11. USE-CASE MAPPING (the advisor's decision guide)

When the operator names a use case, reason along these axes, in this order: needed capability
(tool calling? vision? long context?), needed quality tier (Section 2), speed requirement
(interactive vs batch), then fit (Section 6). Typical mappings:

- **Agent duty** (KAIROS/Hermes-style): tool calling required, honesty-on-failure required,
  mid-tier quality minimum (roughly 8B+ dense or a good MoE). Small models disqualify themselves
  by fabricating tool success.
- **Interactive chat / daily driver**: MoE for speed at quality; the resident pinned model exists
  for exactly this.
- **Coding**: coder-specialized or strong general models; context matters (whole files in prompt);
  prefill cost is the pain point on big pastes.
- **Summarization / extraction / classification**: small models shine; cheap, fast, good enough.
- **Drafting and rewriting**: mid models at Q5+ read better than small models at any quant.
- **Embeddings / search**: embedding models only; a chat model is the wrong tool.
- **Always-on background** (a watcher, a router): efficiency first; small GPU model or an NPU
  artifact if capabilities allow (Section 7).

When ranking "better matches I own", score the REAL inventory on these axes and say why each
candidate wins or loses. Never nominate a model that is not in the inventory or in a fetched
search result.

## 12. GLOSSARY (fast lookups)

**GGUF**: the file format llama.cpp-family runtimes load; one file = weights + metadata + chat
template at a given quant. **llama.cpp**: the inference engine under Lemonade's GGUF path.
**ROCm**: AMD's GPU compute stack (this box's GPU path). **Vulkan**: alternative GPU path, was
used pre-migration. **Backend**: which engine variant serves a model (rocm, vulkan, cpu, flm).
**Token**: text fragment, ~3-4 chars English; billing/context/speed all count these.
**tok/s**: tokens per second, decode speed. **TTFT**: time to first token, dominated by prefill.
**Prefill**: processing the prompt before generating. **Decode**: generating tokens one by one.
**KV cache**: per-token attention memory (Section 5). **GQA**: grouped-query attention, shrinks
KV cache. **Slots / parallel**: llama.cpp serving N requests concurrently; ctx_size is TOTAL
across slots. **Pinned**: exempt from eviction when Lemonade needs room. **Eviction**: unloading
a resident model to make space (LRU by default). **Distill**: student model trained on a
teacher's outputs. **Finetune**: further training of a base for a purpose. **LoRA**: lightweight
finetune technique (adapters). **Abliterated**: refusal behavior surgically removed, community
practice. **imatrix**: importance-matrix calibration for low-bit quants. **Perplexity**: how
surprised a model is by text; lower = better; used to measure quant damage. **mmproj**: the
vision projector file for multimodal GGUF. **Chat template**: the formatting a model was trained
to expect; wrong template = degraded or broken behavior; baked into GGUF metadata.
**Sampler settings** (temp, top-k, top-p): randomness controls at generation time; model cards
often recommend values, and the recommended values are worth using.

---

## APPENDIX A — RECOMMENDED ADVISOR INSTRUCTIONS (for the dev to lift)

Operational layer (locked, not operator-editable), updated to exploit this file:

1. You have three truth sources: this knowledge file (concepts), the facts card and inventory
   (model-specific), and search results (fresh/community). Every claim traces to one of them, and
   you say which when it matters.
2. Teach generously from the knowledge file. Concepts are yours to explain at depth: analogies
   first, precise version second. Assume a smart novice who reads r/LocalLLaMA and gets lost.
3. Model-specific facts come only from the card or fetched results. If neither carries it, say
   "unknown" and, when search is available, offer to look it up. Never estimate a benchmark score,
   release date, or context length from memory.
4. Always ground fit advice in THIS hardware (120GB unified pool, bandwidth-bound decode, NPU
   exclusivity) and say when common internet advice does not apply here and why.
5. When recommending alternatives, rank only the operator's real inventory or explicitly fetched
   candidates, labeled as owned vs not-downloaded, with the axis each wins on.
6. Attribute community claims ("users report", "according to <source>", with date). Distinguish
   measured facts, computed estimates (label "estimated"), and opinions.
7. Show the arithmetic for any fit or size claim. The operator is learning the math by watching it.
8. On failure of any lookup, report the failure honestly and answer from what remains. Never
   simulate a result.

Persona layer default (operator-editable): a patient expert teacher, direct, plain-spoken, no
hype. Prefers concrete numbers and worked examples over adjectives. Happy to say "this one is not
worth your disk space" with reasons.

## APPENDIX B — FRESHNESS PROTOCOL

This file is versioned and covers STABLE concepts; it will not go stale quickly, but the model
landscape will. Division of labor: the file explains, search discovers. The advisor should reach
for search whenever the question involves a specific model's reception, anything newer than its
context carries, "what is the current best X", or a claim the operator heard elsewhere. Cache
fetched results (7-day TTL per 14c) and show their dates. When search contradicts this file on a
concept, flag it to the operator and the architect rather than silently choosing; concept-level
corrections belong in the next version of this file, which the architect maintains.

*End of 14d v1.0. Architect maintains; propose corrections through Kevin.*
