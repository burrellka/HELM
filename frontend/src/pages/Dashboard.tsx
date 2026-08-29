/**
 * Phase 1: the read-only dashboard. Handoff §2.
 *
 * This file RENDERS. It does not derive. Every number here was decided by a
 * table-tested pure function in the backend's `domain/` package — the slot
 * math, the memory pool, the crash state. If you find yourself about to divide
 * `ctx_size` by something in this file, stop: that division is Rule 1, it has
 * a test table, and a second copy of it here is exactly how the two drift
 * apart while both look right (review F4).
 *
 * The rules that show up as JSX:
 *   - `npu_percent ?? 'unavailable'` — NEVER `||`. The gauge is always null
 *     today, but if it is ever fixed and legitimately reports 0, `||` would
 *     render that true 0% as "unavailable": a new lie in the slot where we
 *     removed the old one.
 *   - Unknown gets its own look (`.unknown`, the hatched bar) so it can never
 *     be mistaken for a measurement.
 *   - Every write control is absent, not disabled-with-a-tooltip. Phase 1 is
 *     read-only and the UI should not imply otherwise.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ApiError,
  api,
  type ConcurrencyView,
  type CtxDriftRow,
  type Dashboard as DashboardData,
  type Liveness,
  type ModelDetail,
  type ModelRow,
  type Plan,
  type PoolView,
  type ResourceView,
  type SectionError,
  type ThroughputView,
} from '../api/client';
import { Issues, Modal } from '../components/common';
import { ModelDetailView, fmtCtx } from '../components/ModelDetailView';

/** /live does no work upstream (spec §4.1), so 2s is free. */
const LIVE_POLL_MS = 2000;
/** health + system-stats + stats. Deliberately NOT 2s: the dashboard must not
 *  become a load source on the box it is watching. */
const DASHBOARD_POLL_MS = 5000;
/** Running-context drift (§1.12). Its own slow cadence, deliberately OFF the 5s
 *  poll: it costs an extra /slots read, and the running context only changes
 *  when a model reloads — so a gentle 30s is plenty and keeps /slots off the hot
 *  path (the architect's explicit trap). */
const DRIFT_POLL_MS = 30000;
/** "Update available" is a catalog fact that changes only when Lemonade
 *  re-checks HuggingFace — rare. Its own slow cadence, off the 5s poll. */
const UPDATE_POLL_MS = 60000;

export default function Dashboard({ managementEnabled = false, onConfigure, onUpdate, onOpenDetail }: {
  managementEnabled?: boolean;
  /** Jump to the Library focused on this model — that is where the §5.3
   *  settings editor lives, because it needs the catalog record (the trained
   *  context ceiling) that /health alone does not carry. */
  onConfigure?: (modelName: string) => void;
  /** Jump to the Library focused on this model to run the Update flow (the
   *  re-pull, the model card, the confirm all live there). */
  onUpdate?: (modelName: string) => void;
  /** Open the dedicated per-model detail page. */
  onOpenDetail?: (modelName: string) => void;
}) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [live, setLive] = useState<Liveness | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  /** Running-context drift by model name. Fetched on its own slow cadence, not
   *  in the 5s poll. Missing (undefined) means "not read yet", which the row
   *  shows as nothing — it never invents a "no drift". */
  const [drift, setDrift] = useState<Record<string, CtxDriftRow>>({});
  /** Loaded model names Lemonade reports an upstream update for. Only ever
   *  populated from catalog-MATCHED entries (update_available lives on a catalog
   *  row), so an unmatched/unknown model can never get a pill — the architect's
   *  rule holds by construction. Off the 5s poll: it changes rarely. */
  const [updatable, setUpdatable] = useState<Set<string>>(new Set());
  /** A pending destructive action awaiting confirmation. The 409 flow: HELM
   *  returns the plan, we show what it will evict, THEN the operator decides. */
  const [pending, setPending] = useState<{ plan: Plan; run: () => Promise<void> } | null>(null);

  usePoll(useCallback(async () => {
    try {
      setData(await api.dashboard());
      setFetchError(null);
    } catch (e) {
      // HELM's own API failed (not Lemonade's — that arrives as `errors`).
      // Keep the last good payload on screen and say it's stale rather than
      // blanking the page: a stale-but-labelled table beats nothing.
      setFetchError(e instanceof Error ? e.message : String(e));
    }
  }, []), DASHBOARD_POLL_MS);

  usePoll(useCallback(async () => {
    try {
      setLive(await api.live());
    } catch {
      setLive({ alive: false, latency_ms: null, error_kind: 'helm_unreachable',
        error_detail: 'HELM could not be reached from the browser.' });
    }
  }, []), LIVE_POLL_MS);

  usePoll(useCallback(async () => {
    try {
      const { results } = await api.ctxDrift();
      setDrift(Object.fromEntries(results.map((r) => [r.model_name, r])));
    } catch {
      // A drift read failure is not a fleet problem — it just means we can't
      // show the running-context line this cycle. Leave the last read in place
      // rather than blanking it or, worse, implying "no drift".
    }
  }, []), DRIFT_POLL_MS);

  usePoll(useCallback(async () => {
    try {
      // Only downloaded entries can be loaded, and update_available only exists
      // on a catalog-matched row — so this is already scoped to the models that
      // could honestly show a pill.
      const { entries } = await api.library({ downloaded: true });
      setUpdatable(new Set(
        entries.filter((e) => e.loaded && e.update_available).map((e) => e.model_id),
      ));
    } catch {
      // A failed update check just means no pills this cycle. Never guess.
    }
  }, []), UPDATE_POLL_MS);

  /**
   * Run a control action, routing the 409 "needs confirmation" response into a
   * dialog rather than treating it as an error.
   *
   * The refresh afterwards is deliberate: a load or unload changes the fleet,
   * and waiting up to 5s for the next poll to reflect it would leave the
   * operator staring at a table that contradicts what they just did.
   */
  const act = useCallback(async (attempt: (confirm: boolean) => Promise<unknown>) => {
    const run = async () => {
      await attempt(true);
      setPending(null);
      setData(await api.dashboard());
    };
    try {
      await attempt(false);
      setData(await api.dashboard());
    } catch (e) {
      if (e instanceof ApiError && (e.status === 409 || e.status === 422)) {
        setPending({ plan: e.detail as Plan, run });
        return;
      }
      setFetchError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  if (!data && !fetchError) return <p className="stat-sub">Loading the fleet…</p>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <LiveStrip live={live} generatedAt={data?.generated_at} stale={!!fetchError} />
      {fetchError && (
        <Callout tone="bad" title="HELM's own API is not responding">
          {fetchError}. The figures below are the last successful read and are no longer current.
        </Callout>
      )}
      {data && <SectionErrors errors={data.errors} />}
      {data && <PoolWarning pool={data.resources.pool} />}
      {data && <ResourceHeader resources={data.resources} />}
      {data && (
        <ModelsTable models={data.models} managementEnabled={managementEnabled} act={act}
          onConfigure={onConfigure} onUpdate={onUpdate} onOpenDetail={onOpenDetail}
          drift={drift} updatable={updatable} />
      )}
      {data?.throughput && <ThroughputStrip throughput={data.throughput} />}
      {data && <FooterNote version={data.lemonade_version} managementEnabled={managementEnabled} />}

      {pending && (
        <ConfirmDialog
          plan={pending.plan}
          onClose={() => setPending(null)}
          onConfirm={pending.run}
        />
      )}
    </div>
  );
}

/**
 * The gate between a click and an eviction.
 *
 * Shows exactly what the action will do — including which models it will
 * unload — before it happens. `blocked` means HELM refuses outright and there
 * is no confirm button at all.
 */
function ConfirmDialog({ plan, onClose, onConfirm }: {
  plan: Plan; onClose: () => void; onConfirm: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <Modal
      title={plan.blocked ? 'HELM will not do this' : 'Confirm this action'}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          {!plan.blocked && (
            <button
              type="button" className="btn-danger" disabled={busy}
              onClick={async () => { setBusy(true); try { await onConfirm(); } finally { setBusy(false); } }}
            >{busy ? 'Working…' : 'Yes, do it'}</button>
          )}
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {plan.evicts.length > 0 && (
          <Callout tone="bad" title="This will unload models that are in use">
            {plan.evicts.join(', ')}
          </Callout>
        )}
        <Issues issues={plan.issues} />
      </div>
    </Modal>
  );
}

/**
 * Poll on an interval without overlapping runs.
 *
 * The guard is the point: if the box gets slow and a poll takes longer than
 * the interval, a naive setInterval stacks requests on the machine that is
 * already struggling — turning the monitoring tool into part of the incident.
 */
function usePoll(fn: () => Promise<void>, intervalMs: number) {
  const running = useRef(false);
  const saved = useRef(fn);
  saved.current = fn;

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (running.current || cancelled) return;
      running.current = true;
      try { await saved.current(); } finally { running.current = false; }
    };
    void tick();
    const id = setInterval(tick, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [intervalMs]);
}

// ── liveness ────────────────────────────────────────────────────────────────

function LiveStrip({ live, generatedAt, stale }: {
  live: Liveness | null; generatedAt?: string; stale: boolean;
}) {
  const dead = live && !live.alive;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem' }}>
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: '0.375rem',
        color: dead ? 'var(--status-bad)' : 'var(--status-good)', fontWeight: 600,
      }}>
        <span style={{
          width: 8, height: 8, borderRadius: 999,
          background: dead ? 'var(--status-bad)' : 'var(--status-good)',
        }} />
        {dead ? 'Lemonade not responding' : 'Lemonade live'}
      </span>
      {live?.latency_ms != null && (
        <span className="stat-sub tabular">/live {live.latency_ms}ms</span>
      )}
      {dead && live?.error_detail && <span className="stat-sub">{live.error_detail}</span>}
      {generatedAt && (
        <span className="stat-sub" style={{ marginLeft: 'auto' }}>
          {stale ? 'last good read' : 'updated'} {new Date(generatedAt).toLocaleTimeString()}
        </span>
      )}
    </div>
  );
}

// ── errors ──────────────────────────────────────────────────────────────────

/** One upstream read failed. Name WHICH and WHY — "unreachable", "unauthorized"
 *  and "timeout" are three different operator actions (house rule #5). */
function SectionErrors({ errors }: { errors: Record<string, SectionError> }) {
  const entries = Object.entries(errors);
  if (!entries.length) return null;
  return (
    <>
      {entries.map(([section, error]) => (
        <Callout key={section} tone="warn" title={`Could not read ${section.replace(/_/g, ' ')}`}>
          <span className="mono">{error.kind}</span> — {error.detail}
          <div className="stat-sub" style={{ marginTop: '0.25rem' }}>
            Everything else on this page is still a current read.
          </div>
        </Callout>
      ))}
    </>
  );
}

/** The 80GB saga, catchable in one glance. */
function PoolWarning({ pool }: { pool: PoolView }) {
  if (!pool.warning || !pool.warning_detail) return null;
  const hard = pool.warning === 'small_pool_gtt_disabled';
  return (
    <Callout
      tone={hard ? 'bad' : 'warn'}
      title={hard ? 'GPU memory pool is collapsed — enable_dgpu_gtt is off' : 'GPU memory pool is not readable'}
    >
      {pool.warning_detail}
    </Callout>
  );
}

// ── resource header ─────────────────────────────────────────────────────────

function ResourceHeader({ resources }: { resources: ResourceView }) {
  const { pool } = resources;
  return (
    <div className="card">
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '1.25rem',
      }}>
        <Gauge
          label="GPU memory"
          title={pool.source ?? undefined}
          percent={pool.percent}
          value={
            pool.capacity_gb == null
              ? <span className="unknown">unknown pool</span>
              : <>{pool.used_gb?.toFixed(1) ?? '—'} <span className="stat-sub">/ {pool.capacity_gb.toFixed(1)} GB</span></>
          }
        />
        {/* GPU% can be stuck at 100% on gfx1151 (an amdgpu bug). We keep the
            gauge — it is correct where the driver reports honestly — and read
            temp/power locally from hwmon, which the kernel gets right on every
            AMD GPU. When the gauge is provably lying (pinned high while thermals
            say idle) the backend fills gpu_percent_note and it shows below. */}
        <Gauge label="GPU" percent={resources.gpu_percent}
          value={
            <>
              <Percent value={resources.gpu_percent} />
              <GpuThermals tempC={resources.gpu_temp_c} powerW={resources.gpu_power_w} />
            </>
          }
          footnote={resources.gpu_percent_note ?? undefined} />
        <Gauge label="CPU" percent={resources.cpu_percent}
          value={<Percent value={resources.cpu_percent} />} />
        {/* Rule 3. `??` not `||`. The gauge is broken upstream; say so rather
            than draw a confident 0%. */}
        <Gauge
          label="NPU"
          percent={null}
          value={<span className="unknown">{resources.npu_percent ?? 'unavailable'}</span>}
          footnote="Lemonade always reports null for npu_percent — a known broken gauge, not an idle NPU."
        />
        <div>
          <div className="stat-sub">System RAM</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 600 }} className="tabular">
            {resources.system_ram_gb != null
              ? <>{resources.system_ram_gb.toFixed(1)} <span className="stat-sub">GB</span></>
              : <span className="unknown">unknown</span>}
          </div>
          <div className="stat-sub">Host memory, not the GPU pool.</div>
        </div>
      </div>
      <Capacity resources={resources} />
    </div>
  );
}

function Percent({ value }: { value: number | null }) {
  if (value == null) return <span className="unknown">unavailable</span>;
  return <span className="tabular">{value.toFixed(0)}<span className="stat-sub"> %</span></span>;
}

/** GPU temperature and power, read locally from hwmon. Genuinely more useful
 *  than the percentage (idle vs working is unambiguous) and a thermal readout
 *  matters on Strix Halo, where sustained-load thermal reboots are documented.
 *  Renders nothing when HELM isn't co-located with the GPU (both null). */
function GpuThermals({ tempC, powerW }: { tempC: number | null; powerW: number | null }) {
  if (tempC == null && powerW == null) return null;
  const parts: string[] = [];
  if (tempC != null) parts.push(`${tempC.toFixed(0)}°C`);
  if (powerW != null) parts.push(`${powerW.toFixed(0)} W`);
  return (
    <div className="stat-sub tabular" style={{ marginTop: '0.25rem' }}>{parts.join(' · ')}</div>
  );
}

function Gauge({ label, percent, value, footnote, title }: {
  label: string; percent: number | null; value: React.ReactNode;
  footnote?: string; title?: string;
}) {
  const known = percent != null;
  return (
    <div title={title}>
      <div className="stat-sub">{label}</div>
      <div style={{ fontSize: '1.25rem', fontWeight: 600, margin: '0.125rem 0 0.375rem' }}>
        {value}
      </div>
      <div
        className={known ? undefined : 'unknown-bar'}
        style={{
          height: 6, borderRadius: 999, overflow: 'hidden',
          background: known ? 'var(--bg-subtle)' : undefined,
        }}
      >
        {known && (
          <div style={{
            width: `${Math.min(100, Math.max(0, percent))}%`, height: '100%',
            background: percent > 90 ? 'var(--status-warn)' : 'var(--accent-primary)',
          }} />
        )}
      </div>
      {footnote && <div className="stat-sub" style={{ marginTop: '0.375rem' }}>{footnote}</div>}
    </div>
  );
}

/** Rule 6: max_models is PER TYPE. Never render 3/10 as a global cap. */
function Capacity({ resources }: { resources: ResourceView }) {
  const shown = resources.capacity.filter((c) => c.loaded > 0 || c.type === 'llm');
  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'baseline',
      marginTop: '1rem', paddingTop: '0.875rem', borderTop: '1px solid var(--border-color)',
    }}>
      {shown.map((c) => (
        <span key={c.type} style={{ fontSize: '0.875rem' }}>
          <strong className="tabular">{c.loaded}/{c.max ?? '?'}</strong>{' '}
          {c.type.toUpperCase()}
          {c.pinned > 0 && <span className="stat-sub"> · {c.pinned} pinned</span>}
        </span>
      ))}
      <span className="stat-sub" style={{ marginLeft: 'auto' }}>
        Limits are per type, not a single cap across the box.
      </span>
    </div>
  );
}

// ── the models table ────────────────────────────────────────────────────────

type Act = (attempt: (confirm: boolean) => Promise<unknown>) => Promise<void>;

function ModelsTable({ models, managementEnabled, act, onConfigure, onUpdate, onOpenDetail, drift, updatable }: {
  models: ModelRow[]; managementEnabled: boolean; act: Act;
  onConfigure?: (modelName: string) => void;
  onUpdate?: (modelName: string) => void;
  onOpenDetail?: (modelName: string) => void;
  drift: Record<string, CtxDriftRow>;
  updatable: Set<string>;
}) {
  if (!models.length) {
    return (
      <div className="card">
        <strong>No models loaded.</strong>
        <p className="stat-sub" style={{ marginTop: '0.25rem' }}>
          An empty list is also what the router shows mid-incident while it clears
          hardware state after a backend failure. It does not by itself mean lemond restarted.
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
        <thead>
          <tr style={{ background: 'var(--bg-subtle)', textAlign: 'left' }}>
            {['Model', 'Device', 'Backend', 'Concurrency', 'Pinned', 'Status',
              'LRU order', 'Port / PID', ...(managementEnabled ? ['Actions'] : [])].map((h) => (
              <Th key={h}>{h}</Th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <Row key={m.model_name} model={m} managementEnabled={managementEnabled} act={act}
              onConfigure={onConfigure} onUpdate={onUpdate} onOpenDetail={onOpenDetail}
              drift={drift[m.model_name]} updateAvailable={updatable.has(m.model_name)}
              colSpan={managementEnabled ? 9 : 8} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th style={{
      padding: '0.5rem 0.75rem', fontWeight: 600, fontSize: '0.75rem',
      color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.02em',
      borderBottom: '1px solid var(--border-color)', whiteSpace: 'nowrap',
    }}>{children}</th>
  );
}

function Td({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <td title={title} style={{
      padding: '0.625rem 0.75rem', borderBottom: '1px solid var(--border-color)',
      verticalAlign: 'top',
    }}>{children}</td>
  );
}

function Row({ model, managementEnabled, act, onConfigure, onUpdate, onOpenDetail, drift, updateAvailable, colSpan }: {
  model: ModelRow; managementEnabled: boolean; act: Act;
  onConfigure?: (modelName: string) => void;
  onUpdate?: (modelName: string) => void;
  onOpenDetail?: (modelName: string) => void;
  drift?: CtxDriftRow;
  updateAvailable?: boolean;
  colSpan: number;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <>
    <tr>
      <Td title={model.checkpoint ?? undefined}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button
            type="button" onClick={() => setExpanded((e) => !e)}
            aria-expanded={expanded}
            title={expanded ? 'Hide live detail' : 'Show live detail (throughput, running context, ports)'}
            style={{
              cursor: 'pointer', border: 'none', background: 'none', color: 'var(--text-secondary)',
              padding: 0, fontSize: '0.75rem', width: '1rem', lineHeight: 1,
            }}
          >{expanded ? '▾' : '▸'}</button>
          <span style={{ fontWeight: 600 }}>{model.model_name}</span>
          {updateAvailable && <UpdatePill onClick={() => onUpdate?.(model.model_name)} />}
        </div>
        {model.checkpoint && (
          <div className="stat-sub mono" style={{
            maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>{model.checkpoint}</div>
        )}
      </Td>
      <Td><DeviceBadge device={model.device} /></Td>
      <Td><span className="mono" style={{ fontSize: '0.8125rem' }}>{model.backend_label}</span></Td>
      <Td><Concurrency c={model.concurrency} drift={drift} /></Td>
      <Td>
        {model.pinned
          ? <span title="Exempt from LRU eviction. Pinned does NOT mean it reloads after a lemond restart.">
              Yes
            </span>
          : <span className="stat-sub">—</span>}
      </Td>
      <Td><Status model={model} /></Td>
      <Td>
        <Lru model={model} />
      </Td>
      <Td>
        <span className="mono tabular" style={{ fontSize: '0.8125rem' }}>
          {model.port ?? '—'} <span className="stat-sub">/ {model.pid ?? '—'}</span>
        </span>
      </Td>
      {managementEnabled && (
        <Td>
          <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
            {onConfigure && (
              <button
                type="button" className="btn-secondary"
                title="Slots, context, backend and sampling — with what is saved versus what is running."
                onClick={() => onConfigure(model.model_name)}
              >Settings</button>
            )}
            {/* Pin is now a metadata flip via /internal/pin — instant, no
                reload (14c §2.1). So it runs directly rather than through the
                confirm/plan flow the disruptive actions use. */}
            <button
              type="button" className="btn-secondary"
              title={model.pinned
                ? 'Allow the router to evict this under memory pressure. Instant, no reload.'
                : 'Protect from LRU eviction. Instant, no reload. (Pinned does not auto-reload after a lemond restart.)'}
              onClick={() => void act(() => api.pin(model.model_name, !model.pinned))}
            >{model.pinned ? 'Unpin' : 'Pin'}</button>
            <button
              type="button" className="btn-danger"
              title="Requests in flight will fail. Reloading costs 10-20s."
              onClick={() => void act((confirm) => api.unload(model.model_name, confirm))}
            >Unload</button>
          </div>
        </Td>
      )}
    </tr>
    {expanded && (
      <tr>
        <td colSpan={colSpan} style={{
          padding: 0, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-subtle)',
        }}>
          <DetailPanel modelName={model.model_name} onOpenDetail={onOpenDetail} />
        </td>
      </tr>
    )}
    </>
  );
}

/**
 * The live per-model detail, fetched on demand from /detail when a row expands.
 *
 * On demand is the point: /detail costs three upstream reads (/health, /slots,
 * /metrics), so it is NEVER on the 5s poll — it fires only when the operator
 * opens a row. Everything here keeps the house honesty rules: an implausible
 * per-model rate shows "no measurement", a missing throughput series shows
 * "unavailable" (not 0), and the fields Lemonade can't answer per-model are
 * listed by name with the reason rather than quietly dropped.
 */
function DetailPanel({ modelName, onOpenDetail }: {
  modelName: string; onOpenDetail?: (modelName: string) => void;
}) {
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setDetail(null); setError(null);
    api.modelDetail(modelName)
      .then((d) => { if (live) setDetail(d); })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, [modelName]);

  if (error) {
    return <div style={{ padding: '0.875rem 1rem' }}>
      <span className="unknown">Could not load live detail: {error}</span>
    </div>;
  }
  if (!detail) {
    return <div style={{ padding: '0.875rem 1rem' }} className="stat-sub">Reading live detail…</div>;
  }

  return (
    <div style={{ padding: '1rem 1.25rem' }}>
      <ModelDetailView detail={detail} />
      {onOpenDetail && (
        <button
          type="button" className="btn-secondary" style={{ marginTop: '1rem' }}
          onClick={() => onOpenDetail(modelName)}
        >Open full detail page →</button>
      )}
    </div>
  );
}

/**
 * "Update available" — shown ONLY on catalog-matched loaded models (the data
 * behind it comes from a matched catalog row, so an unmatched or unknown model
 * can never reach here). Clicking routes to the Library's Update flow, where the
 * re-pull, the HF model card, and the confirm live — this pill is the signal,
 * not the action, so it never re-pulls straight from the dashboard.
 */
function UpdatePill({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title="Lemonade reports a newer build of this model on HuggingFace. Opens the Update flow in the Library."
      style={{
        cursor: 'pointer', border: '1px solid var(--status-warn)', background: 'var(--status-warn-soft)',
        color: 'var(--status-warn)', borderRadius: 999, padding: '0.0625rem 0.5rem',
        fontSize: '0.6875rem', fontWeight: 700, letterSpacing: '0.02em', whiteSpace: 'nowrap',
      }}
    >update available</button>
  );
}

function DeviceBadge({ device }: { device: string | null }) {
  if (!device) return <span className="unknown">unknown</span>;
  const cls = { gpu: 'badge badge-gpu', npu: 'badge badge-npu', cpu: 'badge badge-cpu' }[device]
    ?? 'badge badge-cpu';
  return <span className={cls}>{device}</span>;
}

/** Renders the pre-derived string. The math lives in the backend, tested. */
function Concurrency({ c, drift }: { c: ConcurrencyView; drift?: CtxDriftRow }) {
  const unknown = c.kind === 'unknown';
  return (
    <div title={[c.note, c.slot_policy && `slot_policy: ${c.slot_policy}`]
      .filter(Boolean).join('\n') || undefined}>
      <span className={unknown ? 'unknown' : 'tabular'} style={{ fontWeight: unknown ? 400 : 500 }}>
        {c.display}
      </span>
      {c.kind === 'slots' && c.per_slot_ctx != null && (
        <div className="stat-sub">context per slot</div>
      )}
      <DriftLine drift={drift} />
    </div>
  );
}

/**
 * The running-context truth, under the computed value (§1.12).
 *
 * The whole point of this line is the ONE case where it must shout: llama.cpp is
 * running a smaller context than HELM computed (a cap under memory pressure).
 * Everything else stays quiet, and that restraint is deliberate — per the
 * architect, `unknown` is the EXPECTED normal for most loaded models (Lemonade's
 * /slots is model-blind and only attributable to one instance). If `unknown`
 * rendered as a warning, the operator would learn to ignore this line, and then
 * miss the real cap when it appears — the F8 permanently-lit-flag trap. So:
 *   - drift  -> loud (warning color, running vs set)
 *   - match  -> a quiet confirmation, no colour
 *   - unknown-> a muted, neutral note, never a caution
 */
function DriftLine({ drift }: { drift?: CtxDriftRow }) {
  if (!drift) return null; // not read yet — never invent a state
  if (drift.state === 'drift') {
    return (
      <div title={drift.detail}
        style={{ color: 'var(--status-warn)', fontWeight: 600, fontSize: '0.75rem', marginTop: '0.25rem' }}>
        ⚠ running {fmtCtx(drift.running_n_ctx)}
        {drift.computed_per_slot != null && (
          <span style={{ fontWeight: 400 }}> (set {fmtCtx(drift.computed_per_slot)})</span>
        )}
      </div>
    );
  }
  if (drift.state === 'match') {
    return (
      <div className="stat-sub" title={drift.detail} style={{ marginTop: '0.25rem' }}>
        running {fmtCtx(drift.running_n_ctx)} ✓
      </div>
    );
  }
  // unknown: quiet and neutral. It is not a problem, so it must not look like one.
  return (
    <div className="stat-sub" title={drift.detail} style={{ marginTop: '0.25rem' }}>
      running ctx: not attributable
    </div>
  );
}

function Status({ model }: { model: ModelRow }) {
  const tone = model.status === 'dead' ? 'var(--status-bad)'
    : model.status === 'not_ready' ? 'var(--status-warn)'
    : 'var(--status-good)';
  return (
    <div title={model.status_detail ?? undefined}>
      <span style={{ color: tone, fontWeight: 600 }}>
        {model.status === 'dead' ? 'DEAD' : model.status === 'not_ready'
          ? (model.raw_status ?? 'not ready') : 'ready'}
      </span>
      {/* The pid change is the crash detector; watchdog_reset only
          corroborates and is false after real crashes (F8). */}
      {model.restarted_now && (
        <div style={{ color: 'var(--status-warn)', fontSize: '0.75rem', fontWeight: 600 }}>
          ⚠ restarted just now
        </div>
      )}
      {!model.restarted_now && model.last_restart_at && (
        <div
          className={model.restart_is_recent ? undefined : 'stat-sub'}
          style={{ fontSize: '0.75rem', color: model.restart_is_recent ? 'var(--status-warn)' : undefined }}
        >
          restarted {new Date(model.last_restart_at).toLocaleTimeString()}
        </div>
      )}
      {model.watchdog_reset && (
        <div className="stat-sub" title="The router's watchdog force-restarted this backend at some point.">
          watchdog fired
        </div>
      )}
    </div>
  );
}

/**
 * The LRU column.
 *
 * `last_use` is a monotonic counter. It has no epoch, and no endpoint reports
 * the clock's current value, so "3m ago" cannot be computed — only guessed.
 * What the column is FOR is eviction order: under capacity pressure the router
 * drops the least recently used UNPINNED model. Rank answers that exactly,
 * and honestly. (Open question for the architect: what unit is last_use?)
 */
function Lru({ model }: { model: ModelRow }) {
  if (model.lru_rank == null) return <span className="unknown">unknown</span>;
  return (
    <div title={`last_use = ${model.last_use} (monotonic router clock, not a wall-clock time)`}>
      <span className="tabular">#{model.lru_rank}</span>
      {model.lru_rank === 1 && <span className="stat-sub"> most recent</span>}
      {model.next_to_evict && (
        <div className="stat-sub" style={{ color: 'var(--status-warn)' }}>
          next evicted under pressure
        </div>
      )}
    </div>
  );
}

// ── throughput ──────────────────────────────────────────────────────────────

function ThroughputStrip({ throughput: t }: { throughput: ThroughputView }) {
  return (
    <div className="card">
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1.25rem',
      }}>
        <Stat
          label="Last request"
          value={
            // The live box returns exactly 1,000,000 tok/s after a 1-token
            // response. Printing that beside "tok/s" would be this dashboard's
            // first confident lie, so it gets the unknown treatment and the
            // reported number moves into the explanation.
            t.rate_implausible
              ? <span className="unknown">no measurement</span>
              : t.tokens_per_second != null
                ? <>{t.tokens_per_second.toFixed(1)} <span className="stat-sub">tok/s</span></>
                : <span className="unknown">unavailable</span>
          }
          footnote={t.rate_note ?? undefined}
        />
        <Stat label="Time to first token"
          value={t.time_to_first_token != null
            ? <>{(t.time_to_first_token * 1000).toFixed(0)} <span className="stat-sub">ms</span></>
            : <span className="unknown">unavailable</span>} />
        <Stat label="Requests (lifetime)"
          value={t.request_count_total ?? <span className="unknown">unavailable</span>} />
        <Stat label="Tokens in / out (lifetime)"
          value={<>{fmt(t.input_tokens_total)} <span className="stat-sub">/</span> {fmt(t.output_tokens_total)}</>} />
      </div>
    </div>
  );
}

function fmt(n: number | null) {
  return n == null ? '—' : n.toLocaleString();
}

function Stat({ label, value, footnote }: {
  label: string; value: React.ReactNode; footnote?: string;
}) {
  return (
    <div>
      <div className="stat-sub">{label}</div>
      <div className="tabular" style={{ fontSize: '1.25rem', fontWeight: 600 }}>{value}</div>
      {footnote && (
        <div className="stat-sub" style={{ marginTop: '0.25rem', lineHeight: 1.4 }}>{footnote}</div>
      )}
    </div>
  );
}

// ── chrome ──────────────────────────────────────────────────────────────────

function FooterNote({ version, managementEnabled }: {
  version: string | null; managementEnabled: boolean;
}) {
  return (
    <p className="stat-sub">
      {managementEnabled
        ? 'Pin and Unload act on the live fleet. Anything with a consequence asks first.'
        : 'Read-only: no admin key is configured, so write controls are hidden rather than shown failing.'}
      {version && <> Lemonade {version}.</>}
    </p>
  );
}

function Callout({ tone, title, children }: {
  tone: 'good' | 'warn' | 'bad'; title: string; children: React.ReactNode;
}) {
  const fg = { good: 'var(--status-good)', warn: 'var(--status-warn)', bad: 'var(--status-bad)' }[tone];
  const bg = {
    good: 'var(--status-good-soft)', warn: 'var(--status-warn-soft)', bad: 'var(--status-bad-soft)',
  }[tone];
  return (
    <div style={{
      background: bg, border: `1px solid ${fg}33`, borderRadius: 8,
      padding: '0.625rem 0.875rem', fontSize: '0.875rem', lineHeight: 1.5,
    }}>
      <strong style={{ color: fg }}>{title}</strong>
      <div style={{ marginTop: '0.25rem' }}>{children}</div>
    </div>
  );
}
