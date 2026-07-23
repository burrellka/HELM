/**
 * The header download indicator. Sits beside the logo and shows model pulls in
 * progress — because a download is server-owned (it survives a browser refresh),
 * but until now there was nothing telling the operator it was happening.
 *
 * Compact by design: up to 3 active pulls inline, each a name + percent, and a
 * "+N" when more are running. Click anything to open the full modal with a
 * progress bar, percent, downloaded/total bytes, and a derived download speed
 * per model.
 *
 * Speed is NOT in Lemonade's payload, so it is derived here from the change in
 * bytes between polls. That is the only honest way to get it, and it is kept in
 * a ref (not state) so measuring it never triggers a render on its own.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type DownloadJob } from '../api/client';
import { Modal, formatBytes } from './common';

const POLL_MS = 2000;
const INLINE_MAX = 3;

interface Speed { bytesPerSec: number | null }

/** Trim Lemonade's namespaced ids to something readable in a chip. */
function shortName(name: string): string {
  return name.replace(/^(model:)?(user\.)?/, '');
}

export default function DownloadIndicator() {
  const [active, setActive] = useState<DownloadJob[]>([]);
  const [open, setOpen] = useState(false);
  // Per-job speed, derived from byte deltas. In a ref so it doesn't drive renders.
  const prev = useRef<Map<string, { bytes: number; t: number }>>(new Map());
  const [speeds, setSpeeds] = useState<Map<string, Speed>>(new Map());

  const poll = useCallback(async () => {
    try {
      const r = await api.pullJobs();
      const jobs = r.active ?? [];
      // Derive speed from the byte delta since this job was last seen.
      const now = Date.now();
      const nextSpeeds = new Map<string, Speed>();
      const nextPrev = new Map<string, { bytes: number; t: number }>();
      for (const j of jobs) {
        const bytes = j.bytes_downloaded ?? 0;
        const before = prev.current.get(j.id);
        if (before && now > before.t && bytes >= before.bytes) {
          nextSpeeds.set(j.id, { bytesPerSec: (bytes - before.bytes) / ((now - before.t) / 1000) });
        } else {
          nextSpeeds.set(j.id, { bytesPerSec: null });   // first sighting: no rate yet
        }
        nextPrev.set(j.id, { bytes, t: now });
      }
      prev.current = nextPrev;
      setSpeeds(nextSpeeds);
      setActive(jobs);
      if (jobs.length === 0) setOpen(false);   // nothing to show the modal for
    } catch {
      // A transient poll failure shouldn't clear a live indicator; leave the
      // last-known state and try again next tick.
    }
  }, []);

  useEffect(() => {
    void poll();
    const id = setInterval(() => void poll(), POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  if (active.length === 0) return null;

  const inline = active.slice(0, INLINE_MAX);
  const overflow = active.length - inline.length;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Show download progress"
        style={{
          display: 'flex', alignItems: 'center', gap: '0.5rem',
          background: 'var(--bg-subtle)', border: '1px solid var(--border-color)',
          borderRadius: 999, padding: '0.25rem 0.625rem', cursor: 'pointer',
          fontSize: '0.75rem', color: 'var(--text-primary)',
        }}
      >
        <Spinner />
        {inline.map((j) => (
          <span key={j.id} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {shortName(j.model_name)}
            </span>
            <span className="mono stat-sub">{pct(j.percent)}</span>
          </span>
        ))}
        {overflow > 0 && (
          <span className="stat-sub" style={{ fontWeight: 600 }}>+{overflow}</span>
        )}
      </button>

      {open && (
        <Modal title={`Downloads — ${active.length} in progress`} onClose={() => setOpen(false)}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
            {active.map((j) => (
              <DownloadRow key={j.id} job={j} speed={speeds.get(j.id)?.bytesPerSec ?? null} />
            ))}
            <p className="stat-sub" style={{ margin: 0 }}>
              Downloads run on the server, so they keep going if you close this tab or reload. Speed
              is measured live from the bytes arriving.
            </p>
          </div>
        </Modal>
      )}
    </>
  );
}

function DownloadRow({ job, speed }: { job: DownloadJob; speed: number | null }) {
  const p = job.percent != null ? Math.max(0, Math.min(100, job.percent)) : null;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '0.5rem' }}>
        <span style={{ fontWeight: 600, fontSize: '0.875rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {shortName(job.model_name)}
        </span>
        <span className="mono stat-sub">{pct(job.percent)}</span>
      </div>
      {/* Progress bar. Indeterminate stripe when percent is unknown (some pulls
          report bytes but not percent early on). */}
      <div style={{
        height: 6, borderRadius: 3, background: 'var(--bg-subtle)', overflow: 'hidden', margin: '0.35rem 0',
      }}>
        <div style={{
          height: '100%', width: p != null ? `${p}%` : '100%',
          background: 'var(--accent-primary)',
          opacity: p != null ? 1 : 0.35,
          transition: 'width 0.4s ease',
        }} />
      </div>
      <div className="stat-sub" style={{ display: 'flex', gap: '0.75rem', fontSize: '0.75rem' }}>
        <span>
          {formatBytes(job.bytes_downloaded)}{job.bytes_total ? ` / ${formatBytes(job.bytes_total)}` : ''}
        </span>
        <span>{speed != null ? `${formatBytes(speed)}/s` : '…'}</span>
        {job.status && job.status !== 'downloading' && job.status !== 'running' && (
          <span style={{ textTransform: 'capitalize' }}>{job.status}</span>
        )}
      </div>
    </div>
  );
}

function pct(percent: number | null): string {
  return percent != null ? `${Math.round(percent)}%` : '—';
}

/** A tiny spinning ring so the indicator reads as "active" at a glance. */
function Spinner() {
  return (
    <span
      className="dl-spinner"
      style={{
        width: 12, height: 12, borderRadius: '50%',
        border: '2px solid var(--border-strong)', borderTopColor: 'var(--accent-primary)',
        display: 'inline-block',
      }}
    />
  );
}
