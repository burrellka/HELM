/**
 * Phase 4: the live log pane. Spec §5.6, review F5.
 *
 * Consumed with `fetch` + ReadableStream via `api.streamLines`, NOT
 * EventSource — EventSource can't POST, can't send headers, and reconnects on
 * its own schedule. All three are wrong here.
 *
 * Keepalive comments (lines starting with ':') are filtered out inside
 * `streamLines`. They exist to stop Cloudflare 524ing a quiet stream and the
 * operator should never see them.
 *
 * Classification happens SERVER-side (`domain.logs.classify`), batched. The
 * rules are order-sensitive — the nuclear option's line contains the word
 * "unloading", so a naive reimplementation here would render a catastrophe as
 * routine housekeeping. One tested implementation, not two.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type ClassifiedLine } from '../api/client';
import { Callout } from '../components/common';

/** Keep the pane bounded. A log tail left open for a day should not become a
 *  memory leak in the browser watching for one. */
const MAX_LINES = 500;

const EVENT_STYLE: Record<ClassifiedLine['event'], { color?: string; bg?: string; label?: string }> = {
  mass_evict: { color: 'var(--status-bad)', bg: 'var(--status-bad-soft)', label: 'EVERYTHING EVICTED' },
  watchdog: { color: 'var(--status-warn)', bg: 'var(--status-warn-soft)', label: 'watchdog' },
  evict: { color: 'var(--status-warn)', label: 'evict' },
  slots: { color: 'var(--accent-text)', bg: 'var(--accent-soft)', label: 'slots' },
  pool: { color: 'var(--accent-text)', bg: 'var(--accent-soft)', label: 'pool' },
  error: { color: 'var(--status-bad)' },
  plain: {},
};

export default function Logs() {
  const [lines, setLines] = useState<ClassifiedLine[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [onlyEvents, setOnlyEvents] = useState(false);
  const [follow, setFollow] = useState(true);

  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);
  // Batch classification: one round trip per burst, not one per line. A busy
  // lemond emits faster than a request-per-line could ever keep up with.
  const pending = useRef<string[]>([]);

  const stop = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    setStreaming(false);
  }, []);

  const start = useCallback(async () => {
    stop();
    const controller = new AbortController();
    abort.current = controller;
    setStreaming(true);
    setError(null);

    const flush = async () => {
      const batch = pending.current.splice(0, pending.current.length);
      if (!batch.length) return;
      try {
        const { lines: classified } = await api.classifyLines(batch);
        setLines((prev) => [...prev, ...classified].slice(-MAX_LINES));
      } catch {
        // Classification is enrichment. If it fails, still show the text —
        // losing the highlighting is annoying, losing the log is useless.
        setLines((prev) => [
          ...prev,
          ...batch.map((text) => ({ text, event: 'plain' as const, note: null })),
        ].slice(-MAX_LINES));
      }
    };

    const timer = setInterval(() => void flush(), 400);
    try {
      for await (const line of api.streamLines('/operations/logs/stream', controller.signal)) {
        // The backend wraps each line as SSE `data: ...`; strip that framing.
        pending.current.push(line.startsWith('data: ') ? line.slice(6) : line);
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      clearInterval(timer);
      await flush();
      setStreaming(false);
    }
  }, [stop]);

  useEffect(() => () => stop(), [stop]);

  useEffect(() => {
    if (follow) bottom.current?.scrollIntoView({ block: 'end' });
  }, [lines, follow]);

  const visible = lines.filter((l) => {
    if (onlyEvents && l.event === 'plain') return false;
    if (filter && !l.text.toLowerCase().includes(filter.toLowerCase())) return false;
    return true;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {error && <Callout tone="bad" title="Log stream failed">{error}</Callout>}

      <div className="card" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <button type="button" className={streaming ? 'btn-secondary' : 'btn-primary'}
          onClick={() => (streaming ? stop() : void start())}>
          {streaming ? 'Stop' : 'Start streaming'}
        </button>
        <input
          value={filter} placeholder="Filter…" onChange={(e) => setFilter(e.target.value)}
          style={{
            fontFamily: 'inherit', fontSize: '0.875rem', padding: '0.375rem 0.5rem',
            border: '1px solid var(--border-strong)', borderRadius: 8, flex: 1, minWidth: 160,
          }}
        />
        <label style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', fontSize: '0.8125rem' }}>
          <input type="checkbox" checked={onlyEvents} onChange={(e) => setOnlyEvents(e.target.checked)} />
          Events only
        </label>
        <label style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', fontSize: '0.8125rem' }}>
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Follow
        </label>
        <button type="button" className="btn-secondary" onClick={() => setLines([])}>Clear</button>
        <span className="stat-sub">{visible.length} lines</span>
      </div>

      <div className="card" style={{ padding: 0, maxHeight: '60vh', overflowY: 'auto' }}>
        {visible.length === 0 && (
          <div style={{ padding: '1rem' }}>
            <span className="stat-sub">
              {streaming ? 'Waiting for output…' : 'Not streaming.'} The pane highlights the four
              events that explain everything else: a backend crash, an eviction (especially the
              mass eviction that follows a backend failure), a load taking effect, and the memory
              pool figure at startup.
            </span>
          </div>
        )}
        {visible.map((line, i) => {
          const style = EVENT_STYLE[line.event];
          return (
            <div
              key={i}
              title={line.note ?? undefined}
              style={{
                display: 'flex', gap: '0.5rem', alignItems: 'baseline',
                padding: '0.1875rem 0.75rem', background: style.bg,
                borderLeft: style.color ? `3px solid ${style.color}` : '3px solid transparent',
              }}
            >
              {style.label && (
                <span style={{
                  fontSize: '0.625rem', fontWeight: 700, color: style.color,
                  textTransform: 'uppercase', whiteSpace: 'nowrap', minWidth: 92,
                }}>{style.label}</span>
              )}
              <code className="mono" style={{
                fontSize: '0.75rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                color: style.color ?? 'var(--text-primary)',
              }}>{line.text}</code>
            </div>
          );
        })}
        <div ref={bottom} />
      </div>

      <p className="stat-sub">
        Streams survive Cloudflare only with a keepalive and unbuffered proxying. If the tail
        works on the LAN but dies after ~100 seconds through the public hostname, NPM is missing
        the custom nginx block — see <code className="mono">infra/npm-mtls-ingress.md</code>.
      </p>
    </div>
  );
}
