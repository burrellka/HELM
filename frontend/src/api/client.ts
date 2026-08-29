/**
 * API client.
 *
 * Ported from URSA-OSCAR's frontend/src/api/client.ts. Two things here are
 * worth more than they look: ApiError (so failures stay distinguishable) and
 * streamLines (so the log tail survives NPM + Cloudflare).
 */

const BASE = '/api/v1';

/** A failure that still knows what kind of failure it was.
 *
 * House rule, learned expensively: never collapse distinct failures into one
 * message. In KAIROS a 401 surfaced as "Network error" and cost hours. HELM's
 * writes fail in genuinely different ways — no admin key vs. a rejected key
 * vs. NPU exclusivity vs. a dead backend vs. a timeout — and the operator
 * needs to tell them apart to know what to DO. Keep status and detail.
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // FormData sets its own multipart Content-Type (with boundary); forcing JSON
  // here would corrupt an upload. Only default to JSON for non-FormData bodies.
  const isForm = typeof FormData !== 'undefined' && init.body instanceof FormData;
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers: isForm
      ? { ...(init.headers || {}) }
      : { 'Content-Type': 'application/json', ...(init.headers || {}) },
    // The session cookie is httpOnly, so it rides automatically — but only
    // if we ask for it. Same-origin is the default; being explicit documents
    // that auth is cookie-based, not a bearer header we manage in JS.
    credentials: 'same-origin',
  });

  if (resp.status === 401) {
    // Session expired or never existed. The app shell watches for this and
    // routes to /login rather than every caller handling it.
    throw new ApiError(401, 'Not authenticated', await safeBody(resp));
  }
  if (!resp.ok) {
    const body = await safeBody(resp);
    const detail =
      body && typeof body === 'object' && 'detail' in (body as object)
        ? (body as { detail: unknown }).detail
        : body;
    throw new ApiError(resp.status, `${init.method || 'GET'} ${path} -> ${resp.status}`, detail);
  }
  if (resp.status === 204 || !resp.headers.get('content-length')) {
    try { return (await resp.json()) as T; } catch { return undefined as T; }
  }
  return (await resp.json()) as T;
}

async function safeBody(resp: Response): Promise<unknown> {
  try { return await resp.json(); } catch { return undefined; }
}

// ── Types the frame actually returns ────────────────────────────────────────

export interface AuthStatus {
  initialized: boolean;
  authenticated: boolean;
  auth_disabled: boolean;
  /** Which signal proved HTTPS. 'none' behind a real proxy means it isn't
   *  forwarding X-Forwarded-Proto — worth warning about. */
  scheme_source: string;
}

export interface UpstreamStatus {
  reachable: boolean;
  /** Gate every write control on this. When false, HIDE the buttons and say
   *  why — don't render an Unload that 401s. */
  management_enabled: boolean;
  lemonade_url: string;
  error_kind?: string | null;
  error_detail?: string | null;
}

export interface SystemConfig {
  version: string;
  auth_enabled: boolean;
  upstream: UpstreamStatus;
}

/** Detected hardware of the box HELM runs against. All fields are read live
 *  from Lemonade's system-info; nothing about a specific machine is hardcoded,
 *  so this reports whatever box the instance is deployed on. */
export interface SystemProfile {
  detected: boolean;
  processor: string | null;
  cpu_cores: number | null;
  cpu_threads: number | null;
  gpu_name: string | null;
  gpu_family: string | null;
  gpu_pool_gb: number | null;
  gpu_pool_source: string | null;
  npu_available: boolean;
  npu_family: string | null;
  npu_tops: number | null;
  npu_power_mode: string | null;
  physical_memory: string | null;
  used_gb: number | null;
  os_version: string | null;
  lemonade_version: string | null;
  storage_path: string | null;
  storage_free_gb: number | null;
  storage_total_gb: number | null;
  backends: { recipe: string; installed: string[]; installable: string[] }[];
}

// ── Phase 1: the dashboard ──────────────────────────────────────────────────
//
// These mirror backend/src/kblocallm/domain/dashboard.py. Every derived value
// arrives DECIDED — `concurrency.display` is a tested string, `pool.percent`
// is null rather than a guess. Do not recompute any of it here: the whole
// point of F4 is one implementation with one test table, and a second
// derivation in TypeScript is how the two drift apart silently.

/** A failure scoped to one section, so the rest of the dashboard survives it.
 *  `kind` is load-bearing: unreachable / unauthorized / timeout need three
 *  different reactions from the operator. Never render just "error". */
export interface SectionError {
  kind: string;
  detail: string;
}

export interface PoolView {
  /** Capacity. null = unknown; render "unknown", NEVER a remembered 120.5. */
  capacity_gb: number | null;
  used_gb: number | null;
  percent: number | null;
  /** Which field this came from. Shown in the tooltip so "where does 120 come
   *  from?" is answerable on screen. */
  source: string | null;
  warning: string | null;
  warning_detail: string | null;
  gtt_enabled: boolean | null;
}

export interface TypeCapacity {
  type: string;
  loaded: number;
  max: number | null;
  pinned: number;
}

export interface ResourceView {
  pool: PoolView;
  gpu_percent: number | null;
  /** GPU temp (°C) / power (W) read locally from hwmon; null when HELM is not
   *  co-located with the GPU. Read correctly even when gpu_percent is stuck. */
  gpu_temp_c: number | null;
  gpu_power_w: number | null;
  /** Set when gpu_percent is provably lying (pinned 100% while thermals idle).
   *  The gauge still renders; this note sits beside it. */
  gpu_percent_note: string | null;
  cpu_percent: number | null;
  /** Rule 3: ALWAYS null on this box. Render with `??`, never `||` — a
   *  legitimately-fixed 0 must not display as "unavailable". */
  npu_percent: number | null;
  system_ram_gb: number | null;
  capacity: TypeCapacity[];
}

export interface ConcurrencyView {
  kind: 'slots' | 'serial' | 'not_applicable' | 'unknown';
  /** Pre-derived and table-tested. Render this string. */
  display: string;
  slots: number | null;
  per_slot_ctx: number | null;
  slots_defaulted: boolean;
  slot_policy: string | null;
  note: string | null;
}

/** Running-context drift (14c §1.12). Tri-state by construction: a `drift` on an
 *  attributable model is the loud signal; `unknown` is the EXPECTED normal for
 *  most loaded models (Lemonade's /slots is model-blind) and must render quiet,
 *  never as a warning. Mirrors backend/api/models.py CtxDriftRow. */
export interface CtxDriftRow {
  model_name: string;
  state: 'match' | 'drift' | 'unknown';
  computed_per_slot: number | null;
  running_n_ctx: number | null;
  detail: string;
}

/** One normalized per-model detail payload (14c §1.12). Assembled on demand
 *  from /health + /slots + /metrics. Mirrors backend/api/models.py DetailOut. */
export interface ModelDetail {
  model_name: string;
  checkpoint: string | null;
  recipe: string | null;
  device: string | null;
  type: string | null;
  loaded: boolean;
  pinned: boolean;
  backend_label: string;
  max_context_window: number | null;
  port: number | null;
  pid: number | null;
  concurrency: { kind: string; display: string; slots: number | null; per_slot_ctx: number | null; note: string | null };
  ctx_drift: CtxDriftRow;
  /** Per-model throughput from /metrics. null = no series (unavailable, not 0). */
  throughput: {
    tokens_per_second: number | null;
    rate_implausible: boolean;
    rate_note: string | null;
    time_to_first_token: number | null;
    requests_total: number | null;
    input_tokens_total: number | null;
    output_tokens_total: number | null;
  } | null;
  /** §1.12 fields Lemonade can't answer per-model yet — named, not omitted. */
  deferred: { name: string; reason: string }[];
}

export interface ModelRow {
  model_name: string;
  checkpoint: string | null;
  device: string | null;
  type: string | null;
  recipe: string | null;
  llamacpp_backend: string | null;
  backend_label: string;
  concurrency: ConcurrencyView;
  pinned: boolean;
  max_context_window: number | null;

  status: 'ready' | 'dead' | 'not_ready';
  raw_status: string | null;
  status_detail: string | null;
  restarted_now: boolean;
  last_restart_at: string | null;
  restart_is_recent: boolean;
  watchdog_reset: boolean;

  /** The raw LRU clock: monotonic, unit unconfirmed, no epoch. Not a time. */
  last_use: number | null;
  /** 1 = most recently used. What the LRU column is actually for. */
  lru_rank: number | null;
  next_to_evict: boolean;

  pid: number | null;
  port: number | null;
  backend_url: string | null;
}

export interface ThroughputView {
  tokens_per_second: number | null;
  time_to_first_token: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  request_count_total: number | null;
  input_tokens_total: number | null;
  output_tokens_total: number | null;
  /** True when the reported rate is an artifact (the live box returns
   *  1,000,000 tok/s after a 1-token response). Show the unknown treatment. */
  rate_implausible: boolean;
  rate_note: string | null;
}

export interface Dashboard {
  generated_at: string;
  lemonade_version: string | null;
  reachable: boolean;
  management_enabled: boolean;
  models: ModelRow[];
  resources: ResourceView;
  throughput: ThroughputView | null;
  errors: Record<string, SectionError>;
}

export interface Liveness {
  alive: boolean;
  latency_ms: number | null;
  error_kind: string | null;
  error_detail: string | null;
}

// ── Phase 2: control ────────────────────────────────────────────────────────

export interface PlanIssue {
  level: 'error' | 'warning' | 'info';
  code: string;
  message: string;
}

export interface Plan {
  blocked: boolean;
  /** True when there are warnings the operator hasn't acknowledged. The UI
   *  shows them and asks; it does not proceed. */
  needs_confirmation: boolean;
  issues: PlanIssue[];
  /** Models this action will unload. NPU exclusivity. */
  evicts: string[];
  /** The total HELM computed from slots x ctx_per_slot. Shown as a
   *  consequence, never asked of the operator. */
  computed_ctx_size: number | null;
  /** Exactly what would be POSTed upstream. A control plane the operator
   *  can't audit is one they're right not to trust. */
  body: Record<string, unknown>;
}

export interface ActionResult {
  performed: boolean;
  plan: Plan;
  result?: unknown;
  error_kind?: string | null;
  error_detail?: string | null;
  /** Read-back of a load's OPTIONS. `performed` only means Lemonade accepted the
   *  reload; `verified === false` means it came back running DIFFERENT options
   *  than were sent (e.g. a dropped --parallel). null = nothing to verify. */
  verified?: boolean | null;
  verify_note?: string | null;
}

export interface LoadRequest {
  model_name: string;
  recipe?: string | null;
  slots?: number | null;
  /** PER SLOT. The backend multiplies. */
  ctx_per_slot?: number | null;
  llamacpp_backend?: string | null;
  pinned?: boolean;
  /** Sampling params. Omitted/null = leave unset (model default). The backend
   *  composes these into llamacpp_args; never type the flags in extra_args. */
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  min_p?: number | null;
  repeat_penalty?: number | null;
  extra_args?: string;
  save_options?: boolean;
}

// ── Phase 3: library ────────────────────────────────────────────────────────

export interface DriftField {
  field: string;
  kind: 'same' | 'drift' | 'not_saved' | 'not_running' | 'unknown';
  saved: unknown;
  running: unknown;
  /** Expected and not worth alarm (Lemonade auto-detects llamacpp_backend). */
  benign: boolean;
  note: string | null;
}

export interface Drift {
  has_drift: boolean;
  has_unpersisted: boolean;
  fields: DriftField[];
}

/** One side (saved or running) of a model's config, in the operator's units.
 *  `ctx_per_slot` is DERIVED SERVER-SIDE — never divide ctx_size here. */
export interface ModelSettings {
  slots: number | null;
  ctx_per_slot: number | null;
  /** The raw total, for reference. Never an input; showing it as one is the
   *  trap Rule 1 exists to prevent. */
  ctx_size: number | null;
  backend: string | null;
  /** Sampling params lifted out of llamacpp_args into their own fields.
   *  null = the flag was absent (leave unset). */
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  min_p: number | null;
  repeat_penalty: number | null;
  /** llamacpp_args with --parallel and the sampling flags already stripped by
   *  the shared parser. */
  extra_args: string;
}

export interface LibraryEntry {
  model_id: string;
  recipe: string | null;
  checkpoint: string | null;
  downloaded: boolean;
  suggested: boolean;
  update_available: boolean;
  size: number | null;
  max_context_window: number | null;
  labels: string[];
  raw_labels: string[];
  /** matched = badges are real. no_labels = genuinely none (an ANSWER).
   *  unknown = the join failed; render "capabilities unknown", NEVER an empty
   *  badge row — the two look identical to a reader and mean opposite things. */
  label_join: 'matched' | 'no_labels' | 'unknown';
  /** Rule 10: wiring an agent to this model fails SILENTLY. */
  strips_tools_silently: boolean;
  loaded: boolean;
  pinned: boolean;
  device: string | null;
  description: string;
  favorite: boolean;
  archived: boolean;
  tags: string[];
  drift: Drift | null;
  /** What is persisted. */
  saved_settings: ModelSettings | null;
  /** What is in memory right now. Null unless loaded. */
  running_settings: ModelSettings | null;
}

export interface Library {
  entries: LibraryEntry[];
  known_tags: string[];
  total: number;
  /** Loaded models with no catalog row: the F9 join failing in the wild. */
  unmatched_loaded: string[];
}

/** One HuggingFace search hit (a downloadable GGUF model not necessarily in
 *  your catalog). Mirrors backend/api/library.py HFHit. */
export interface HFHit {
  id: string;
  /** Downloads in the LAST 30 DAYS (HuggingFace's `downloads` field). */
  downloads: number | null;
  likes: number | null;
  updated: string | null;
  url: string;
  /** True when last-month downloads clear the popularity bar (🔥). */
  popular: boolean;
  suggested_name: string;
}

/** One GGUF file (quant) in an HF repo, with a grounded fit against your pool
 *  and everything needed to pull exactly this quant. */
export interface HFFile {
  filename: string;
  quant: string | null;
  size_gb: number | null;
  fit_verdict: string | null;
  fit_detail: string | null;
  pull_name: string;
  checkpoint: string;
  recipe: string;
}

/** Pre-download read of an HF model: its own README + per-quant size/fit. */
export interface HFDetail {
  repo: string;
  card: string | null;
  card_note: string | null;
  files: HFFile[];
  pool_gb: number | null;
  used_gb: number | null;
}

export interface LibraryQuery {
  text?: string;
  labels?: string;
  tags?: string;
  recipes?: string;
  downloaded?: boolean;
  favorite?: boolean;
  loaded?: boolean;
  include_archived?: boolean;
}

export interface MetadataPatch {
  description?: string;
  favorite?: boolean;
  archived?: boolean;
  tags?: string[];
}

// ── Phase 4: operations ─────────────────────────────────────────────────────

export interface ClassifiedLine {
  text: string;
  event: 'watchdog' | 'evict' | 'mass_evict' | 'slots' | 'pool' | 'error' | 'plain';
  note: string | null;
}

export interface BenchmarkRun {
  model_id: string;
  scenario: string;
  started_at: string;
  tokens_per_second: number | null;
  time_to_first_token: number | null;
  output_tokens: number | null;
  ok: number;
  error: string;
  config_note: string;
}

/** Build a query string, dropping empty/undefined so the backend sees "not
 *  specified" rather than an empty filter that matches nothing. */
// ── config surface types (14c 2.3) ──────────────────────────────────────────

export interface ConfigIssue {
  level: 'error' | 'danger' | 'warning' | 'info';
  code: string;
  message: string;
  key: string | null;
}

export interface ConfigChange {
  key: string;
  old: unknown;
  new: unknown;
  kind: 'added' | 'removed' | 'changed';
}

export interface ConfigValidation {
  blocked: boolean;
  /** True when a dangerous key (enable_dgpu_gtt, ports, auth) is being changed.
   *  The UI must make the operator acknowledge it explicitly. */
  needs_danger_ack: boolean;
  dangerous_keys: string[];
  issues: ConfigIssue[];
  changes: ConfigChange[];
}

export interface ModelConfigRead {
  model_id: string;
  recipe: string | null;
  max_context_window: number | null;
  running_text: string;
  saved_text: string;
}

export interface ConfigBackup {
  id: number;
  reason: string;
  created_at: string;
  preview: string;
}

// ── Model Advisor types (14c Part 1) ─────────────────────────────────────────

export interface SavedSummary {
  text: string;
  analysis_model: string | null;
  generated_at: string;
  saved: true;
}

export interface AnalyzeResult {
  model_id: string;
  /** Layer 1: deterministic local facts. Always present, never hallucinated. */
  local_facts: Record<string, any>;
  /** Layer 2: HF metadata, or an honest "no external record". */
  external: Record<string, any>;
  /** Layer 3: generated FRESH this call. Null unless synthesize was asked. */
  synthesis: Record<string, any> | null;
  /** A previously-saved fast Summary, shown instantly so slow generation isn't
   *  repeated. Null if none saved. */
  saved_summary: SavedSummary | null;
  /** A previously-saved Deep analysis — its own slot, so it doesn't collide with
   *  the saved Summary. Null if none saved. */
  saved_deep: SavedSummary | null;
}

/** One event off the streaming advisor chat. `text` and `reasoning` are deltas
 *  (append as they arrive); `complete` is terminal with the per-turn numbers;
 *  `error` carries a distinct failure kind, never a generic "unavailable". */
export interface AdvisorChatEvent {
  event: 'text' | 'reasoning' | 'complete' | 'error';
  text?: string;
  model?: string;
  grounded_on?: string;
  elapsed_s?: number;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number;
            prompt_tokens_details?: { cached_tokens?: number } };
  note?: string;
  kind?: string;
  detail?: string;
}

/** A generated block from Summary (§1.10) or the deep synthesis. `available`
 *  false carries `note` (why it couldn't) instead of `text`. */
export interface GeneratedResult {
  generated: true;
  available: boolean;
  model: string;
  text?: string;
  label?: string;
  note?: string;
  finish_reason?: string | null;
}

/** One in-progress model download, normalized by the backend from Lemonade's
 *  downloads endpoint. `percent` is 0-100; speed is NOT here — the client
 *  derives it from byte deltas between polls. */
export interface DownloadJob {
  id: string;
  model_name: string;
  percent: number | null;
  bytes_downloaded: number | null;
  bytes_total: number | null;
  status: string | null;
}

export interface DownloadJobs {
  /** Every job Lemonade knows about (includes finished/failed). */
  jobs: unknown[];
  /** Just the ones downloading right now. */
  active: DownloadJob[];
  active_count: number;
}

export interface AdvisorModelOption {
  id: string;
  params_b: number | null;
  /** True for sub-2B: fabricates in this role, never the silent default. */
  not_recommended: boolean;
  warning: string | null;
}

export interface AdvisorSettings {
  settings: {
    analysis_model: string;
    analysis_model_is_default: boolean;
    /** The full editable instruction set (UI label: "Advisor instructions").
     *  Defaults to the architect's shipped instructions.txt. */
    persona_prompt: string;
    persona_is_default: boolean;
    /** The knowledge file, injected as authoritative concept grounding.
     *  Defaults to the architect's shipped 14d. */
    knowledge_primer: string;
    knowledge_primer_is_default: boolean;
    knowledge_primer_active: boolean;
    /** Editable search provider + model (the key stays in .env). */
    web_search_provider: string;
    web_search_model: string;
    web_search_model_is_default: boolean;
    /** Summary (§1.10): its own model + instruction box, independent of Analyze.
     *  Defaults to Aevum-0.6B and the shipped Appendix S instructions. */
    summary_model: string;
    summary_model_is_default: boolean;
    summary_prompt: string;
    summary_prompt_is_default: boolean;
  };
  analysis_models: AdvisorModelOption[];
  /** The Summary picker: same owned models, small-first, without the analyst's
   *  "sub-2B fabricates" warning (a small model is right for Summary). */
  summary_models: AdvisorModelOption[];
  operational_prompt_locked: boolean;
  /** Search status. `enabled` reflects the key presence; provider/model mirror
   *  the editable settings above. The key is never returned. */
  web_search: { enabled: boolean; provider: string; model: string };
}

function qs(params: object): string {
  const search = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') search.set(k, String(v));
  }
  const s = search.toString();
  return s ? `?${s}` : '';
}

export const api = {
  authStatus: () => request<AuthStatus>('/auth/status'),
  setup: (password: string) =>
    request<{ ok: boolean }>('/auth/setup', {
      method: 'POST', body: JSON.stringify({ password }),
    }),
  login: (password: string) =>
    request<{ ok: boolean }>('/auth/login', {
      method: 'POST', body: JSON.stringify({ password }),
    }),
  logout: () => request<{ ok: boolean }>('/auth/logout', { method: 'POST' }),
  systemConfig: () => request<SystemConfig>('/system/config'),
  /** The DETECTED hardware of the box HELM is pointed at — live, not hardcoded,
   *  so it reflects whatever machine this instance runs against. */
  systemProfile: () => request<SystemProfile>('/system/profile'),

  /** The ~5s poll. Always 200; per-section failures arrive in `errors`. */
  dashboard: () => request<Dashboard>('/dashboard'),
  /** The 2s poll. /live does no work upstream, so it is safe to hammer. */
  live: () => request<Liveness>('/live'),

  /** Running-context drift for all loaded llamacpp models, one /slots read.
   *  DELIBERATELY off the dashboard poll (§1.12: /slots stays off the always-on
   *  cadence) — the Dashboard fetches this on a gentle, separate interval. */
  ctxDrift: () => request<{ results: CtxDriftRow[] }>('/models/ctx-drift'),

  /** Full per-model detail (identity + concurrency + drift + throughput +
   *  honestly-named unavailable fields). On demand only — 3 upstream reads. */
  modelDetail: (modelId: string) =>
    request<ModelDetail>(`/models/${encodeURIComponent(modelId)}/detail`),

  // ── control ───────────────────────────────────────────────────────────────

  /** Dry run. Performs nothing. Call it as the dialog changes so the operator
   *  sees "this will unload qwen3.5-2b-FLM" BEFORE the button, not after. */
  planLoad: (body: LoadRequest) =>
    request<Plan>('/control/plan/load', { method: 'POST', body: JSON.stringify(body) }),

  /** Throws ApiError 409 with a Plan in `detail` when confirmation is needed,
   *  422 when blocked. Both are expected flows, not bugs. */
  load: (body: LoadRequest, confirm = false) =>
    request<ActionResult>(`/control/load${qs({ confirm })}`, {
      method: 'POST', body: JSON.stringify(body),
    }),

  unload: (model_name: string, confirm = false) =>
    request<ActionResult>(`/control/unload${qs({ confirm })}`, {
      method: 'POST', body: JSON.stringify({ model_name }),
    }),

  /** Instant metadata flip via /internal/pin — no reload, no confirm gate. */
  pin: (model_name: string, pinned: boolean) =>
    request<ActionResult>('/control/pin', {
      method: 'POST', body: JSON.stringify({ model_name, pinned }),
    }),

  // ── library ───────────────────────────────────────────────────────────────

  library: (query: LibraryQuery = {}) => request<Library>(`/library${qs(query)}`),

  /** Live HuggingFace GGUF search — the discovery Lemonade's UI has and HELM's
   *  catalog-only search lacks. Queries HF directly (no Lemonade proxy exists). */
  hfSearch: (query: string, sort: 'downloads' | 'lastModified' = 'downloads') =>
    request<{ query: string; results: HFHit[]; error: string | null }>(
      `/library/hf-search${qs({ query, sort })}`),

  /** Pre-download read for an HF repo: its README + per-quant size and a
   *  grounded fit against your detected pool (the Advisor's own fit math). */
  hfDetail: (repo: string) => request<HFDetail>(`/library/hf-detail${qs({ repo })}`),

  updateMetadata: (modelId: string, patch: MetadataPatch) =>
    request<Record<string, unknown>>(`/library/${encodeURIComponent(modelId)}/metadata`, {
      method: 'PATCH', body: JSON.stringify(patch),
    }),

  exportMetadata: () => request<unknown>('/library/export'),

  pull: (model_name: string, checkpoint?: string, recipe?: string) =>
    request<{ started: boolean; warnings: string[] }>('/library/pull', {
      method: 'POST', body: JSON.stringify({ model_name, checkpoint, recipe }),
    }),

  /** Server-owned download jobs — survives a browser refresh. `active` is the
   *  normalized "downloading right now" view the header indicator polls. */
  pullJobs: () => request<DownloadJobs>('/library/jobs'),

  /** The HuggingFace model card (README) for a model, so the operator can see
   *  what it is before downloading/updating. `card` is null when there's no HF
   *  repo (a custom/local model) — never an invented one. */
  modelCard: (modelId: string) =>
    request<{ repo_id: string | null; card: string | null; note: string | null }>(
      `/library/card/${encodeURIComponent(modelId)}`),

  deleteModel: (model_name: string, confirm = false) =>
    request<{ deleted: boolean }>('/library/delete', {
      method: 'POST', body: JSON.stringify({ model_name, confirm }),
    }),

  // ── operations ────────────────────────────────────────────────────────────

  classifyLines: (lines: string[]) =>
    request<{ lines: ClassifiedLine[] }>('/operations/logs/classify', {
      method: 'POST', body: JSON.stringify({ lines }),
    }),

  warmSet: () => request<{ models: string[]; note: string }>('/operations/warm-set'),
  addWarm: (model_id: string) =>
    request<{ models: string[] }>('/operations/warm-set', {
      method: 'POST', body: JSON.stringify({ model_id }),
    }),
  removeWarm: (model_id: string) =>
    request<{ models: string[] }>(`/operations/warm-set/${encodeURIComponent(model_id)}`, {
      method: 'DELETE',
    }),
  warm: (confirm = false) =>
    request<{ warmed: unknown[]; skipped: unknown[] }>(`/operations/warm${qs({ confirm })}`, {
      method: 'POST',
    }),

  benchmarks: (model_id?: string) =>
    request<{ runs: BenchmarkRun[] }>(`/operations/benchmarks${qs({ model_id })}`),
  benchmark: (model_name: string, scenario: string) =>
    request<unknown>('/operations/benchmark', {
      method: 'POST', body: JSON.stringify({ model_name, scenario }),
    }),

  // ── config surface (14c 2.3): read-first, validate, diff, backup, restore ───

  readModelConfig: (modelId: string) =>
    request<ModelConfigRead>(`/config/model/${encodeURIComponent(modelId)}`),
  validateModelConfig: (modelId: string, text: string) =>
    request<ConfigValidation>(`/config/model/${encodeURIComponent(modelId)}/validate`, {
      method: 'POST', body: JSON.stringify({ text }),
    }),
  applyModelConfig: (modelId: string, text: string, confirm = false) =>
    request<{ applied: boolean; backup_id?: number; verified?: boolean; error_kind?: string; error_detail?: string }>(
      `/config/model/${encodeURIComponent(modelId)}/apply${qs({ confirm })}`,
      { method: 'POST', body: JSON.stringify({ text }) }),
  resetModelConfig: (modelId: string, confirm = false) =>
    request<{ applied: boolean; cleared?: boolean; note?: string | null }>(
      `/config/model/${encodeURIComponent(modelId)}/reset${qs({ confirm })}`, { method: 'POST' }),

  readGlobalConfig: () => request<{ text: string; current: Record<string, unknown> }>('/config/global'),
  validateGlobalConfig: (text: string, current: Record<string, unknown>) =>
    request<ConfigValidation>('/config/global/validate', {
      method: 'POST', body: JSON.stringify({ text, current }),
    }),
  applyGlobalConfig: (text: string) =>
    request<unknown>('/config/global/apply', { method: 'POST', body: JSON.stringify({ text }) }),

  configBackups: (scope: string) =>
    request<{ backups: ConfigBackup[] }>(`/config/backups${qs({ scope })}`),
  restoreBackup: (backupId: number, confirm = false) =>
    request<{ restored: boolean }>(`/config/backups/${backupId}/restore${qs({ confirm })}`, {
      method: 'POST',
    }),

  // ── Model Advisor (14c Part 1): grounded, never recalled ────────────────────

  /** Layer 1 facts render instantly; pass synthesize to run Layer 3 (the model
   *  over the facts). Layer 2 (HF) is cached server-side. */
  analyze: (modelId: string, synthesize = false) =>
    request<AnalyzeResult>(`/advisor/analyze/${encodeURIComponent(modelId)}${qs({ synthesize })}`, {
      method: 'POST',
    }),
  /** Summary (§1.10): the fast, starved-context read. Its own model (Aevum by
   *  default), facts card only, low token cap. Seconds, not minutes. */
  summarize: (modelId: string) =>
    request<GeneratedResult>(`/advisor/summarize/${encodeURIComponent(modelId)}`, {
      method: 'POST',
    }),
  saveSummary: (modelId: string, text: string, analysisModel: string | null,
                kind: 'summary' | 'deep' = 'summary') =>
    request<{ saved: boolean; generated_at: string }>(
      `/advisor/summary/${encodeURIComponent(modelId)}`, {
        method: 'POST', body: JSON.stringify({ text, analysis_model: analysisModel, kind }),
      }),
  deleteSummary: (modelId: string, kind: 'summary' | 'deep' = 'summary') =>
    request<{ deleted: boolean }>(
      `/advisor/summary/${encodeURIComponent(modelId)}${qs({ kind })}`, {
        method: 'DELETE',
      }),
  advisorSettings: () => request<AdvisorSettings>('/advisor/settings'),
  updateAdvisorSettings: (patch: {
    analysis_model?: string; persona_prompt?: string; knowledge_primer?: string;
    web_search_provider?: string; web_search_model?: string;
    summary_model?: string; summary_prompt?: string;
  }) =>
    request<AdvisorSettings['settings']>('/advisor/settings', {
      method: 'PUT', body: JSON.stringify(patch),
    }),
  testWebSearch: () =>
    request<{ ok: boolean; error: string | null; sample: string | null; source_count: number }>(
      '/advisor/web-search/test', { method: 'POST' }),
  /** Attach a knowledge file by upload rather than paste. */
  uploadKnowledgeFile: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<{ attached: boolean; chars: number }>('/advisor/knowledge-file', {
      method: 'POST', body: form,
    });
  },
  advisorChat: (modelId: string, message: string, history: { role: string; content: string }[]) =>
    request<{ reply: string; model: string; grounded_on: string }>('/advisor/chat', {
      method: 'POST', body: JSON.stringify({ model_id: modelId, message, history }),
    }),

  /**
   * Streaming advisor chat — the UI path. Yields parsed events as they arrive so
   * the answer forms live instead of landing after a ~1-2 min silence. The 26B
   * reasons before it answers, so expect `reasoning` events first (render as a
   * live 'thinking' indicator, NEVER as the answer), then `text` deltas, then a
   * terminal `complete` carrying usage + wall-clock for the per-turn line.
   */
  streamAdvisorChat: async function* (
    modelId: string, message: string, history: { role: string; content: string }[],
    signal?: AbortSignal,
  ): AsyncGenerator<AdvisorChatEvent, void, void> {
    const resp = await fetch(`${BASE}/advisor/chat/stream`, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ model_id: modelId, message, history }),
      signal,
    });
    if (!resp.ok || !resp.body) {
      throw new ApiError(resp.status, `advisor chat stream -> ${resp.status}`, await safeBody(resp));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    const emit = function* (line: string): Generator<AdvisorChatEvent> {
      if (!line || line.startsWith(':')) return;          // keepalive comment
      const data = line.startsWith('data:') ? line.slice(5).trim() : line;
      if (!data) return;
      try { yield JSON.parse(data) as AdvisorChatEvent; } catch { /* skip a partial frame */ }
    };
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          yield* emit(buffer.trim());
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl).replace(/\r$/, '');
          buffer = buffer.slice(nl + 1);
          yield* emit(line);
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  /**
   * Read a streaming endpoint line by line (the log tail, spec §5.6).
   *
   * `fetch` + ReadableStream, NOT EventSource. EventSource can't POST, can't
   * set headers, and reconnects on its own schedule — all three are wrong
   * here.
   *
   * Buffer across reads: one read() is NOT one line. A chunk can split a
   * line in half and it will, at exactly the moment a demo is happening.
   *
   * Lines beginning with ':' are keepalive comments from the backend and
   * MUST be skipped silently — they exist to stop Cloudflare 524ing a quiet
   * stream, and the user should never see them.
   */
  streamLines: async function* (
    path: string,
    signal?: AbortSignal,
  ): AsyncGenerator<string, void, void> {
    const resp = await fetch(`${BASE}${path}`, {
      credentials: 'same-origin',
      headers: { Accept: 'text/event-stream' },
      signal,
    });
    if (!resp.ok || !resp.body) {
      throw new ApiError(resp.status, `stream ${path} -> ${resp.status}`, await safeBody(resp));
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          // Flush whatever's left; the last line often has no trailing \n.
          buffer += decoder.decode();
          const tail = buffer.trim();
          if (tail && !tail.startsWith(':')) yield tail;
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl).replace(/\r$/, '');
          buffer = buffer.slice(nl + 1);
          if (!line || line.startsWith(':')) continue;  // keepalive
          yield line;
        }
      }
    } finally {
      reader.releaseLock();
    }
  },
};
