/**
 * The Analyze panel. Spec 14c Part 1, §1.0-1.5, §1.10.
 *
 * The one principle this renders: grounded, never recalled. Every claim carries
 * its provenance, and the sources look different on purpose so a reader can
 * never mistake a generated sentence for a measured fact:
 *
 *   - Layer 1 (local facts) — deterministic, computed, no model ran.
 *   - Layer 2 (external metadata) — HuggingFace, or an explicit "no external
 *     record". Never a fabricated card.
 *   - Summary (§1.10) — a FAST, plain-language read from the facts card by a
 *     small model (Aevum by default). Seconds, not minutes. Saveable.
 *   - Deep analysis (Layer 3) — the full teaching write-up by the big analysis
 *     model over facts + inventory + web search. Slower, opt-in.
 *
 * Summary and Deep analysis are two DIFFERENT operations, not one button at two
 * verbosities: the live verify showed the old single synthesis was
 * generation-bound (~2 min on the 26B), so Summary runs its own small model
 * over a starved context to hit a couple of seconds. See advisor_service.
 */
import { useCallback, useEffect, useState } from 'react';
import { api, type AnalyzeResult, type GeneratedResult } from '../api/client';
import { Badge, Callout, Unknown } from './common';
import Markdown from './Markdown';

/**
 * A generated block collapsed to a few lines by default so it does not dominate
 * the panel. Expand to read it all, collapse back. The clamp is a max-height on
 * the collapsed state; the markdown underneath renders identically either way.
 */
function CollapsibleMarkdown({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const longEnough = text.length > 220 || text.split('\n').length > 4;
  return (
    <div>
      <div style={{
        maxHeight: open || !longEnough ? 'none' : '4.6rem',
        overflow: open || !longEnough ? 'visible' : 'hidden',
        position: 'relative',
        maskImage: open || !longEnough ? 'none'
          : 'linear-gradient(to bottom, black 60%, transparent 100%)',
        WebkitMaskImage: open || !longEnough ? 'none'
          : 'linear-gradient(to bottom, black 60%, transparent 100%)',
      }}>
        <Markdown>{text}</Markdown>
      </div>
      {longEnough && (
        <button type="button" onClick={() => setOpen((o) => !o)}
          style={{
            marginTop: '0.375rem', background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--accent-primary)', fontSize: '0.8125rem', fontWeight: 600, padding: 0,
          }}>
          {open ? 'Show less ▲' : 'Show more ▼'}
        </button>
      )}
    </div>
  );
}

export default function AnalyzePanel({ modelId }: { modelId: string }) {
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Summary (fast) state.
  const [summarizing, setSummarizing] = useState(false);
  const [saving, setSaving] = useState(false);
  /** A freshly generated summary not yet saved. Held so the operator can review,
   *  then Save it, without confusing it with the persisted one. */
  const [freshSummary, setFreshSummary] = useState<GeneratedResult | null>(null);

  // Deep analysis (heavy) state — separate, opt-in, not persisted.
  const [deepRunning, setDeepRunning] = useState(false);
  const [deep, setDeep] = useState<Record<string, any> | null>(null);

  const loadFacts = useCallback(async () => {
    setError(null);
    try {
      // Layer 1 + Layer 2 + any saved summary. No model runs.
      setResult(await api.analyze(modelId, false));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [modelId]);

  const runSummary = useCallback(async () => {
    setSummarizing(true); setError(null);
    try {
      const r = await api.summarize(modelId);
      setFreshSummary(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSummarizing(false);
    }
  }, [modelId]);

  const saveFresh = useCallback(async () => {
    if (!freshSummary?.text) return;
    setSaving(true);
    try {
      await api.saveSummary(modelId, freshSummary.text, freshSummary.model ?? null);
      setFreshSummary(null);
      await loadFacts();  // reload so it shows as the saved summary
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [freshSummary, modelId, loadFacts]);

  const runDeep = useCallback(async () => {
    setDeepRunning(true); setError(null);
    try {
      const r = await api.analyze(modelId, true);
      setDeep(r.synthesis);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeepRunning(false);
    }
  }, [modelId]);

  useEffect(() => { void loadFacts(); }, [loadFacts]);

  if (error) return <Callout tone="bad" title="Analyze failed">{error}</Callout>;
  if (!result) return <p className="stat-sub">Reading the facts…</p>;

  const f = result.local_facts;
  const ext = result.external;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Layer 1 */}
      <section>
        <ProvenanceHeader
          tone="good"
          label="Local facts"
          detail="Computed from the model file and catalog. No model ran; nothing here is generated." />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '0.875rem', marginTop: '0.625rem' }}>
          <Fact label="Parameters" value={
            f.parameters
              ? <>{f.parameters.total_b}B{f.parameters.moe && <span className="stat-sub"> total · {f.parameters.active_b}B active (MoE)</span>}</>
              : <Unknown>not stated</Unknown>} />
          <Fact label="Device / recipe" value={<>{(f.device ?? '?').toUpperCase()} <span className="stat-sub">/ {f.recipe ?? '?'}</span></>} />
          <Fact label="Trained context" value={f.max_context_window ? f.max_context_window.toLocaleString() : <Unknown>unknown</Unknown>} />
          <Fact label="Size on disk" value={f.size_gb != null ? `${f.size_gb} GB` : <Unknown>n/a (NPU)</Unknown>} />
        </div>

        {f.quant && (
          <div className="card" style={{ marginTop: '0.75rem', background: 'var(--bg-subtle)' }}>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <Badge tone="accent">{f.quant.label}</Badge>
              <span className="stat-sub">{f.quant.bytes_per_weight ? `${f.quant.bytes_per_weight} bytes/weight` : 'size figure unknown'}</span>
            </div>
            <p style={{ fontSize: '0.875rem', margin: '0.5rem 0 0' }}>{f.quant.explanation}</p>
            {f.quant.size_impact && <p className="stat-sub" style={{ marginTop: '0.375rem' }}>{f.quant.size_impact}</p>}
            {f.quant.tradeoff && <p className="stat-sub" style={{ marginTop: '0.375rem' }}>{f.quant.tradeoff}</p>}
          </div>
        )}

        {f.fit && (
          <Callout tone={f.fit.verdict === 'too_big' ? 'bad' : f.fit.verdict === 'fits_if_evict' ? 'warn' : 'good'}
            title={{
              fits_resident: 'Fits alongside what is loaded',
              fits_if_evict: 'Loads only if something is evicted',
              too_big: 'Will not fit',
              unknown: 'Fit unknown',
            }[f.fit.verdict as string] ?? 'Fit'}>
            {f.fit.detail}
            {f.fit.footprint_gb != null && !f.fit.measured && (
              <span className="stat-sub"> (estimated from on-disk size)</span>)}
          </Callout>
        )}

        <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
          {(f.capability_labels ?? []).map((l: string) => (
            <Badge key={l} tone={l === 'tool-calling' ? 'good' : 'accent'}>{l}</Badge>
          ))}
          {(f.capability_labels ?? []).length === 0 && <Unknown>no capability labels</Unknown>}
        </div>
      </section>

      {/* Layer 2 */}
      <section>
        <ProvenanceHeader
          tone="accent"
          label="External metadata"
          detail="HuggingFace, cached. Structured signal from a known source, or an honest absence." />
        {ext.status === 'found' && ext.metadata ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.75rem', marginTop: '0.625rem' }}>
            <Fact label="Repo" value={<span className="mono" style={{ fontSize: '0.8125rem' }}>{ext.repo_id}</span>} />
            <Fact label="Downloads" value={ext.metadata.downloads?.toLocaleString() ?? <Unknown />} />
            <Fact label="Likes" value={ext.metadata.likes ?? <Unknown />} />
            <Fact label="License" value={ext.metadata.license ?? <Unknown>unknown</Unknown>} />
          </div>
        ) : (
          <Callout tone="info">
            <strong>No external record.</strong> {ext.resolution || ext.note}
          </Callout>
        )}
      </section>

      {/* Summary (fast, §1.10) */}
      <section>
        <ProvenanceHeader
          tone="warn"
          label="Summary"
          detail="A quick plain-language read of the facts above, by a small fast model. Seconds, not minutes. Save it to keep it." />
        <Summary
          fresh={freshSummary}
          saved={result.saved_summary}
          summarizing={summarizing}
          saving={saving}
          onSummarize={() => void runSummary()}
          onSave={() => void saveFresh()}
        />
      </section>

      {/* Deep analysis (heavy Layer 3, opt-in) */}
      <section>
        <ProvenanceHeader
          tone="warn"
          label="Deep analysis"
          detail="The full teaching write-up: what it's good for, fit math, alternatives from your own models, live web search. Runs the big analysis model — slower." />
        <DeepAnalysis
          result={deep}
          running={deepRunning}
          onRun={() => void runDeep()}
        />
      </section>
    </div>
  );
}

/**
 * The fast Summary block. Kevin's flow, now on the small model:
 *   - a saved summary shows instantly (no model run), with Regenerate;
 *   - Summarize/Regenerate runs the small model (a couple of seconds), then
 *     offers Save;
 *   - Save persists it; next open it is instant again.
 */
function Summary({ fresh, saved, summarizing, saving, onSummarize, onSave }: {
  fresh: GeneratedResult | null;
  saved: AnalyzeResult['saved_summary'];
  summarizing: boolean; saving: boolean;
  onSummarize: () => void; onSave: () => void;
}) {
  const btnLabel = summarizing ? 'Summarizing…' : (saved || fresh) ? 'Regenerate' : 'Summarize';

  // A just-generated summary awaiting Save takes priority in the view.
  if (fresh) {
    if (!fresh.available) {
      return (
        <div style={{ marginTop: '0.5rem' }}>
          <Callout tone="info">{fresh.note}</Callout>
          <button type="button" className="btn-secondary" style={{ marginTop: '0.5rem' }}
            disabled={summarizing} onClick={onSummarize}>{btnLabel}</button>
        </div>
      );
    }
    return (
      <div style={{ marginTop: '0.625rem' }}>
        <div className="card" style={{ borderLeft: '3px solid var(--status-warn)' }}>
          <CollapsibleMarkdown text={fresh.text ?? ''} />
          {fresh.label && <p className="stat-sub" style={{ marginTop: '0.5rem' }}>{fresh.label}</p>}
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.5rem' }}>
          <button type="button" className="btn-primary" disabled={saving} onClick={onSave}>
            {saving ? 'Saving…' : 'Save this summary'}
          </button>
          <button type="button" className="btn-secondary" disabled={summarizing} onClick={onSummarize}>
            {btnLabel}
          </button>
          <span className="stat-sub">Not saved yet. Save it to see it here instantly next time.</span>
        </div>
      </div>
    );
  }

  if (saved) {
    return (
      <div style={{ marginTop: '0.625rem' }}>
        <div className="card" style={{ borderLeft: '3px solid var(--status-warn)' }}>
          <CollapsibleMarkdown text={saved.text} />
          <p className="stat-sub" style={{ marginTop: '0.5rem' }}>
            Saved summary{saved.analysis_model ? ` · by ${saved.analysis_model}` : ''}
            {saved.generated_at ? ` · ${new Date(saved.generated_at + 'Z').toLocaleString()}` : ''}.
            A dated snapshot over the facts as they were; the facts above are always live.
          </p>
        </div>
        <div style={{ marginTop: '0.5rem' }}>
          <button type="button" className="btn-secondary" disabled={summarizing} onClick={onSummarize}>
            {btnLabel}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <button type="button" className="btn-secondary" disabled={summarizing} onClick={onSummarize}>
        {btnLabel}
      </button>
      <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
        A quick "what is this, in plain words" from the facts above. Save it once you like it and it
        shows instantly next time. For the full teaching write-up, use Deep analysis below.
      </p>
    </div>
  );
}

/**
 * The heavy synthesis, opt-in and generate-on-demand (not persisted — Summary is
 * the saved snapshot). This is the deep teach the architect kept unchanged.
 */
function DeepAnalysis({ result, running, onRun }: {
  result: Record<string, any> | null; running: boolean; onRun: () => void;
}) {
  const btnLabel = running ? 'Analyzing…' : result ? 'Regenerate deep analysis' : 'Run deep analysis (the big model, slower)';

  if (result && !result.available) {
    return (
      <div style={{ marginTop: '0.5rem' }}>
        <Callout tone="info">{String(result.note ?? '')}</Callout>
        <button type="button" className="btn-secondary" style={{ marginTop: '0.5rem' }}
          disabled={running} onClick={onRun}>{btnLabel}</button>
      </div>
    );
  }

  if (result?.available) {
    return (
      <div style={{ marginTop: '0.625rem' }}>
        <div className="card" style={{ borderLeft: '3px solid var(--status-warn)' }}>
          <CollapsibleMarkdown text={String(result.text ?? '')} />
          {result.label && <p className="stat-sub" style={{ marginTop: '0.5rem' }}>{String(result.label)}</p>}
        </div>
        <div style={{ marginTop: '0.5rem' }}>
          <button type="button" className="btn-secondary" disabled={running} onClick={onRun}>{btnLabel}</button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <button type="button" className="btn-secondary" disabled={running} onClick={onRun}>{btnLabel}</button>
      <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
        Reads the facts, your other models, and (if a search key is set) the live web, then teaches
        the whole picture. This is the slow one; the Summary above is the quick read.
      </p>
    </div>
  );
}

function ProvenanceHeader({ tone, label, detail }: {
  tone: 'good' | 'accent' | 'warn'; label: string; detail: string;
}) {
  const color = { good: 'var(--status-good)', accent: 'var(--accent-text)', warn: 'var(--status-warn)' }[tone];
  return (
    <div>
      <span style={{
        fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em',
        color, borderBottom: `2px solid ${color}`, paddingBottom: 2,
      }}>{label}</span>
      <p className="stat-sub" style={{ margin: '0.375rem 0 0' }}>{detail}</p>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="stat-sub">{label}</div>
      <div style={{ fontSize: '1rem', fontWeight: 600 }}>{value}</div>
    </div>
  );
}
