/**
 * The body of a model's live detail (14c §1.12) — shared by the dashboard's
 * expandable row and the dedicated detail page so the two never drift apart.
 *
 * Every honesty rule the rest of HELM follows shows up here as JSX:
 *   - an implausible per-model rate (the box's literal 1,000,000 tok/s) renders
 *     "no measurement", never a printed number;
 *   - a missing throughput series renders "unavailable", never 0;
 *   - the drift line is quiet for `unknown` (the expected normal) and only loud
 *     for a real cap on an attributable model;
 *   - the fields Lemonade can't answer per-model are listed BY NAME with the
 *     reason, not silently dropped.
 */
import type { ModelDetail } from '../api/client';

/** 65536 -> '64K'; non-round numbers keep their digits. Mirrors the backend's
 *  format_ctx so both surfaces read identically. */
export function fmtCtx(n: number | null): string {
  if (n == null) return '—';
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : n.toLocaleString();
}

function fmt(n: number | null): string {
  return n == null ? '—' : n.toLocaleString();
}

export function ModelDetailView({ detail }: { detail: ModelDetail }) {
  const t = detail.throughput;
  const d = detail.ctx_drift;
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '1.5rem',
    }}>
      <Block title="Throughput (this model)">
        <Stat label="Tokens / second" value={
          !t ? <span className="unknown">unavailable</span>
            : t.rate_implausible ? <span className="unknown" title={t.rate_note ?? undefined}>no measurement</span>
            : t.tokens_per_second != null ? <>{t.tokens_per_second.toFixed(1)} <span className="stat-sub">tok/s</span></>
            : <span className="unknown">unavailable</span>
        } />
        <Stat label="Time to first token" value={
          t?.time_to_first_token != null
            ? <>{(t.time_to_first_token * 1000).toFixed(0)} <span className="stat-sub">ms</span></>
            : <span className="unknown">unavailable</span>
        } />
        <Stat label="Requests (lifetime)" value={
          t?.requests_total != null ? t.requests_total.toLocaleString() : <span className="unknown">unavailable</span>
        } />
        <Stat label="Tokens in / out (lifetime)" value={
          t ? <>{fmt(t.input_tokens_total)} <span className="stat-sub">/</span> {fmt(t.output_tokens_total)}</>
            : <span className="unknown">unavailable</span>
        } />
      </Block>

      <Block title="Running context">
        <Stat label="Computed per slot" value={
          detail.concurrency.per_slot_ctx != null
            ? fmtCtx(detail.concurrency.per_slot_ctx) : <span className="unknown">unknown</span>
        } />
        <Stat label="Actually running" value={
          d.state === 'unknown'
            ? <span className="unknown" title={d.detail}>not attributable</span>
            : <span title={d.detail}
                style={d.state === 'drift' ? { color: 'var(--status-warn)', fontWeight: 600 } : undefined}>
                {fmtCtx(d.running_n_ctx)}{d.state === 'drift' ? ' ⚠' : ' ✓'}
              </span>
        } />
        <Stat label="Concurrency" value={<span className="tabular">{detail.concurrency.display}</span>} />
        <Stat label="Max context (trained)" value={
          detail.max_context_window != null ? fmtCtx(detail.max_context_window) : <span className="unknown">unknown</span>
        } />
      </Block>

      <Block title="Identity">
        <Stat label="Backend" value={<span className="mono">{detail.backend_label}</span>} />
        <Stat label="Device" value={detail.device ?? <span className="unknown">unknown</span>} />
        <Stat label="Port / PID" value={
          <span className="mono tabular">{detail.port ?? '—'} <span className="stat-sub">/ {detail.pid ?? '—'}</span></span>
        } />
        <Stat label="Pinned" value={detail.pinned ? 'Yes' : <span className="stat-sub">no</span>} />
      </Block>

      {detail.deferred.length > 0 && (
        <Block title="Not available per model">
          {detail.deferred.map((f) => (
            <div key={f.name} style={{ marginBottom: '0.6rem' }}>
              <div style={{ fontSize: '0.8125rem', fontWeight: 500 }}>
                {f.name}: <span className="unknown">unavailable</span>
              </div>
              <div className="stat-sub" style={{ lineHeight: 1.4 }}>{f.reason}</div>
            </div>
          ))}
        </Block>
      )}
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="stat-sub" style={{
        textTransform: 'uppercase', letterSpacing: '0.02em', fontWeight: 600, marginBottom: '0.6rem',
      }}>{title}</div>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', marginBottom: '0.4rem', fontSize: '0.875rem' }}>
      <span className="stat-sub">{label}</span>
      <span className="tabular" style={{ textAlign: 'right' }}>{value}</span>
    </div>
  );
}
