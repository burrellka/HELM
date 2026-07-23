/**
 * Phase 4: warm-on-startup and benchmarks. Spec Rule 5, §5.7.
 *
 * The warm set is the interesting one. Rule 5 identifies a real gap — pinned
 * does NOT mean auto-load, so after `systemctl restart lemond` everything is
 * cold even though it still says pinned. But automating the fix is the most
 * dangerous idea in the spec: there is no restart signal in the API, and the
 * only available inference (an empty model list) is ALSO what the router shows
 * mid-incident while it clears hardware state after a backend failure.
 *
 * So this page is a button a human presses, having looked at the box. The UI
 * says so rather than hiding it behind a toggle that implies otherwise.
 */
import { useCallback, useEffect, useState } from 'react';
import { ApiError, api, type BenchmarkRun, type ConfigValidation, type SystemProfile } from '../api/client';
import { Callout, Unknown } from '../components/common';

export default function Operations({ managementEnabled }: { managementEnabled: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <CurrentSystemSetup />
      <WarmSet managementEnabled={managementEnabled} />
      <ServerConfig />
      <Benchmarks />
    </div>
  );
}

/**
 * Current system setup — the DETECTED hardware of the box HELM runs against.
 *
 * Everything here is read live from Lemonade's system-info, not hardcoded.
 * That is the point: HELM is deployable to any box (Kevin is adding more Strix
 * Halo machines), and this panel plus the advisor both reason about whatever
 * machine the instance is pointed at. Deploy elsewhere and this reports that
 * box's specs, and the advisor's fit/speed advice follows.
 */
function CurrentSystemSetup() {
  const [p, setP] = useState<SystemProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.systemProfile().then(setP).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <h2 style={{ fontSize: '1rem' }}>Current system setup</h2>
      <p className="stat-sub" style={{ maxWidth: 760, marginTop: 0 }}>
        Detected live from this box, not hardcoded. HELM and the advisor reason about the machine
        it is actually pointed at, so the same image runs correctly on any Strix Halo you deploy it
        to.
      </p>
      {error && <Callout tone="bad" title="Could not read system info">{error}</Callout>}
      {p && !p.detected && (
        <Callout tone="warn" title="Hardware not detected">
          HELM could not read this box's hardware from Lemonade. Nothing is assumed.
        </Callout>
      )}
      {p && p.detected && (
        <div className="card">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1rem' }}>
            <Spec label="Processor" value={p.processor} sub={p.cpu_cores != null ? `${p.cpu_cores} cores / ${p.cpu_threads} threads` : undefined} />
            <Spec label="GPU" value={p.gpu_name || p.gpu_family} sub={p.gpu_family ?? undefined} />
            <Spec
              label="GPU memory pool"
              value={p.gpu_pool_gb != null ? `${p.gpu_pool_gb.toFixed(1)} GB` : null}
              sub={p.used_gb != null ? `${p.used_gb.toFixed(1)} GB in use now` : undefined}
              title={p.gpu_pool_source ?? undefined} />
            <Spec label="NPU" value={p.npu_available ? `${p.npu_family}${p.npu_tops ? ` · ~${p.npu_tops} TOPS` : ''}` : 'none detected'}
              sub={p.npu_available && p.npu_power_mode ? `power mode ${p.npu_power_mode}` : undefined} />
            <Spec label="Physical memory" value={p.physical_memory} />
            <Spec label="Model storage" value={p.storage_total_gb != null ? `${p.storage_free_gb} GB free of ${p.storage_total_gb} GB` : null} sub={p.storage_path ?? undefined} />
            <Spec label="OS" value={p.os_version} />
            <Spec label="Lemonade" value={p.lemonade_version} />
          </div>
          {p.backends.length > 0 && (
            <div style={{ marginTop: '1rem', paddingTop: '0.875rem', borderTop: '1px solid var(--border-color)' }}>
              <div className="stat-sub" style={{ marginBottom: '0.5rem' }}>Inference backends on this box</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {p.backends.map((b) => (
                  <span key={b.recipe} style={{ fontSize: '0.8125rem' }}>
                    <strong className="mono">{b.recipe}</strong>
                    {b.installed.length > 0 && <span style={{ color: 'var(--status-good)' }}> {b.installed.join(', ')}</span>}
                    {b.installable.length > 0 && <span className="stat-sub"> (+{b.installable.join('/')})</span>}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function Spec({ label, value, sub, title }: {
  label: string; value: string | null | undefined; sub?: string; title?: string;
}) {
  return (
    <div title={title}>
      <div className="stat-sub">{label}</div>
      <div style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
        {value ? value : <Unknown>unknown</Unknown>}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/**
 * Level B: the global config.json surface. Spec §5.3b, 14c §2.3.
 *
 * Read and validate are live: it loads the real config, diffs edits, and flags
 * dangerous keys (enable_dgpu_gtt, ports, auth) with a named warning. Apply is
 * deliberately NOT wired, because the write mechanism is unconfirmed on this
 * Lemonade build. The UI says exactly that rather than presenting a button
 * that silently fails or, worse, half-writes a config that bricks the server.
 */
function ServerConfig() {
  const [current, setCurrent] = useState<Record<string, unknown> | null>(null);
  const [text, setText] = useState('');
  const [validation, setValidation] = useState<ConfigValidation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    api.readGlobalConfig().then((r) => { setCurrent(r.current); setText(r.text); })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open]);

  useEffect(() => {
    if (!current) return;
    const id = setTimeout(() => {
      api.validateGlobalConfig(text, current).then(setValidation).catch(() => setValidation(null));
    }, 300);
    return () => clearTimeout(id);
  }, [text, current]);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <h2 style={{ fontSize: '1rem' }}>Server config (config.json)</h2>
      {!open ? (
        <div>
          <button type="button" className="btn-secondary" onClick={() => setOpen(true)}>
            View server config
          </button>
          <p className="stat-sub" style={{ marginTop: '0.375rem', maxWidth: 720 }}>
            The global Lemonade config: the memory pool switch, ports, backends. High blast radius.
            Editing here is read-and-validate for now; applying a global change is not yet wired,
            because a global edit is assumed to need a full server restart that drops every model.
          </p>
        </div>
      ) : (
        <>
          {error && <Callout tone="bad" title="Could not read config">{error}</Callout>}
          <Callout tone="warn" title="Global config, whole-fleet blast radius">
            A change here is assumed to require a full Lemonade restart, which drops all models and
            interrupts every agent. HELM reads, validates and diffs it, but does not apply global
            changes yet: the write mechanism is unconfirmed on this build. Change it on the box with
            <code className="mono"> lemonade config set</code> until that is traced.
          </Callout>
          <textarea
            value={text} spellCheck={false} rows={14} className="mono"
            onChange={(e) => setText(e.target.value)}
            style={{
              fontSize: '0.75rem', padding: '0.625rem', borderRadius: 8,
              border: `1px solid ${validation?.blocked ? 'var(--status-bad)' : 'var(--border-strong)'}`,
              background: 'var(--bg-subtle)', resize: 'vertical', lineHeight: 1.5,
            }}
          />
          {validation?.issues.map((iss, i) => (
            <Callout key={i} tone={iss.level === 'error' ? 'bad'
              : iss.level === 'danger' ? 'bad' : iss.level === 'warning' ? 'warn' : 'info'}>
              {iss.level === 'danger' && <strong>⚠ Dangerous change. </strong>}
              {iss.key && <code className="mono">{iss.key}</code>}{iss.key && ': '}{iss.message}
            </Callout>
          ))}
          {validation && validation.changes.length > 0 && (
            <div className="stat-sub">
              {validation.changes.length} change(s) staged. Apply on the box, then re-read here.
            </div>
          )}
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button type="button" className="btn-secondary" onClick={() => setOpen(false)}>Close</button>
            <button type="button" className="btn-primary" disabled
              title="Global config apply is not wired: the write mechanism is unconfirmed on this build.">
              Apply (not available yet)
            </button>
          </div>
        </>
      )}
    </section>
  );
}

function WarmSet({ managementEnabled }: { managementEnabled: boolean }) {
  const [models, setModels] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [adding, setAdding] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [pending, setPending] = useState<{ message: string; plan: unknown[] } | null>(null);

  const refresh = useCallback(async () => {
    const data = await api.warmSet();
    setModels(data.models);
    setNote(data.note);
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const run = async (confirm: boolean) => {
    setResult(null);
    try {
      const data = await api.warm(confirm);
      setPending(null);
      setResult(
        `Warmed ${data.warmed.length}, skipped ${data.skipped.length}. ` +
        JSON.stringify(data.skipped),
      );
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        // Expected: needs confirmation, or an NPU model was skipped.
        const detail = e.detail as { message: string; plan?: unknown[]; skipped?: unknown[] };
        setPending({ message: detail.message, plan: detail.plan ?? [] });
        if (detail.skipped?.length) {
          setResult(`Will skip: ${JSON.stringify(detail.skipped)}`);
        }
        return;
      }
      setResult(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <h2 style={{ fontSize: '1rem' }}>Warm on startup</h2>

      <Callout tone="info" title="Why this is a button and not a setting">
        {note || 'Loading…'}
      </Callout>

      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
        {models.length === 0 && (
          <span className="stat-sub">
            The warm set is empty. Add models explicitly — HELM will not warm "everything that
            was pinned", because pinned means "do not evict", which is a different intent from
            "bring this back after a restart".
          </span>
        )}
        {models.map((m) => (
          <div key={m} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="mono" style={{ fontSize: '0.875rem' }}>{m}</span>
            <button type="button" className="btn-secondary"
              onClick={async () => { await api.removeWarm(m); void refresh(); }}>Remove</button>
          </div>
        ))}
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            value={adding} placeholder="Model id to add to the warm set"
            onChange={(e) => setAdding(e.target.value)}
            style={{
              flex: 1, fontFamily: 'inherit', fontSize: '0.875rem', padding: '0.375rem 0.5rem',
              border: '1px solid var(--border-strong)', borderRadius: 8,
            }}
          />
          <button type="button" className="btn-secondary" disabled={!adding}
            onClick={async () => { await api.addWarm(adding); setAdding(''); void refresh(); }}>
            Add
          </button>
        </div>
      </div>

      {pending && (
        <Callout tone="warn" title="Confirm before warming">
          {pending.message}
          <div style={{ marginTop: '0.5rem' }}>
            <button type="button" className="btn-danger" onClick={() => void run(true)}>
              The box is idle — warm now
            </button>
          </div>
        </Callout>
      )}

      {result && <Callout tone="info">{result}</Callout>}

      {managementEnabled && (
        <div>
          <button type="button" className="btn-primary" disabled={!models.length}
            onClick={() => void run(false)}>
            Warm the fleet now
          </button>
        </div>
      )}
    </section>
  );
}

function Benchmarks() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);

  useEffect(() => {
    api.benchmarks().then((d) => setRuns(d.runs)).catch(() => setRuns([]));
  }, []);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <h2 style={{ fontSize: '1rem' }}>Benchmarks</h2>
      <p className="stat-sub" style={{ maxWidth: 720 }}>
        Stored so model choices are evidence-based rather than folklore. Each result records the
        configuration it was measured under — a tok/s figure without its slot count and context
        is not comparable to the next one. Runs are started from a model's row in the library and
        execute real inference on the shared box, so they are never scheduled or automatic.
      </p>

      {runs.length === 0 ? (
        <div className="card"><span className="stat-sub">No benchmark runs recorded yet.</span></div>
      ) : (
        <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ background: 'var(--bg-subtle)', textAlign: 'left' }}>
                {['Model', 'Scenario', 'When', 'tok/s', 'Tokens', 'Configuration'].map((h) => (
                  <th key={h} style={{
                    padding: '0.5rem 0.75rem', fontSize: '0.75rem', fontWeight: 600,
                    color: 'var(--text-secondary)', textTransform: 'uppercase',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={i}>
                  <td style={cell}>{r.model_id}</td>
                  <td style={cell}>{r.scenario}</td>
                  <td style={cell}>{new Date(r.started_at).toLocaleString()}</td>
                  <td style={cell} className="tabular">
                    {r.ok
                      ? (r.tokens_per_second?.toFixed(1) ?? <span className="unknown">—</span>)
                      : <span style={{ color: 'var(--status-bad)' }}>failed</span>}
                  </td>
                  <td style={cell} className="tabular">{r.output_tokens ?? '—'}</td>
                  <td style={cell}>
                    <code className="mono stat-sub">{r.config_note}</code>
                    {r.error && <div className="stat-sub" style={{ color: 'var(--status-bad)' }}>{r.error}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

const cell: React.CSSProperties = {
  padding: '0.5rem 0.75rem', borderBottom: '1px solid var(--border-color)', verticalAlign: 'top',
};
