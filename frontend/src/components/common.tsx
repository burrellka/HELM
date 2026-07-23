/**
 * Shared primitives. Extracted once the Dashboard stopped being the only page.
 *
 * `Callout` and `Unknown` carry rules, not just styling:
 *   - a failure must name WHICH failure it was (house rule #5);
 *   - "we don't know" must never look like a value (Rule 3, F9).
 */
import { useState } from 'react';
import type { PlanIssue } from '../api/client';

export function Callout({ tone, title, children }: {
  tone: 'good' | 'warn' | 'bad' | 'info';
  title?: string;
  children: React.ReactNode;
}) {
  const fg = {
    good: 'var(--status-good)', warn: 'var(--status-warn)',
    bad: 'var(--status-bad)', info: 'var(--accent-text)',
  }[tone];
  const bg = {
    good: 'var(--status-good-soft)', warn: 'var(--status-warn-soft)',
    bad: 'var(--status-bad-soft)', info: 'var(--accent-soft)',
  }[tone];
  return (
    <div style={{
      background: bg, border: `1px solid ${fg}33`, borderRadius: 8,
      padding: '0.625rem 0.875rem', fontSize: '0.875rem', lineHeight: 1.5,
    }}>
      {title && <strong style={{ color: fg }}>{title}</strong>}
      <div style={{ marginTop: title ? '0.25rem' : 0 }}>{children}</div>
    </div>
  );
}

/** The honest-ignorance treatment. Never renders as a number or an empty slot. */
export function Unknown({ children = 'unknown' }: { children?: React.ReactNode }) {
  return <span className="unknown">{children}</span>;
}

export function Badge({ tone = 'neutral', title, children }: {
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'accent';
  title?: string;
  children: React.ReactNode;
}) {
  const palette = {
    neutral: ['var(--status-neutral-soft)', 'var(--status-neutral)'],
    good: ['var(--status-good-soft)', 'var(--status-good)'],
    warn: ['var(--status-warn-soft)', 'var(--status-warn)'],
    bad: ['var(--status-bad-soft)', 'var(--status-bad)'],
    accent: ['var(--accent-soft)', 'var(--accent-text)'],
  }[tone];
  return (
    <span className="badge" title={title} style={{ background: palette[0], color: palette[1] }}>
      {children}
    </span>
  );
}

/** Renders a plan's issues. Errors first — they're why the button is disabled. */
export function Issues({ issues }: { issues: PlanIssue[] }) {
  const order = { error: 0, warning: 1, info: 2 } as const;
  const sorted = [...issues].sort((a, b) => order[a.level] - order[b.level]);
  if (!sorted.length) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {sorted.map((issue, i) => (
        <Callout
          key={`${issue.code}-${i}`}
          tone={issue.level === 'error' ? 'bad' : issue.level === 'warning' ? 'warn' : 'info'}
        >
          {issue.message}
        </Callout>
      ))}
    </div>
  );
}

export function Modal({ title, onClose, children, footer }: {
  title: string; onClose: () => void;
  children: React.ReactNode; footer?: React.ReactNode;
}) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.35)',
        display: 'grid', placeItems: 'center', padding: '1rem', zIndex: 50,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-elevated)', borderRadius: 12, boxShadow: 'var(--shadow-lg)',
          width: 'min(680px, 100%)', maxHeight: '85vh', display: 'flex', flexDirection: 'column',
        }}
      >
        <header style={{
          padding: '0.875rem 1rem', borderBottom: '1px solid var(--border-color)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <strong>{title}</strong>
          <button type="button" className="btn-secondary" onClick={onClose}>Close</button>
        </header>
        <div style={{ padding: '1rem', overflowY: 'auto' }}>{children}</div>
        {footer && (
          <footer style={{
            padding: '0.75rem 1rem', borderTop: '1px solid var(--border-color)',
            display: 'flex', gap: '0.5rem', justifyContent: 'flex-end',
          }}>{footer}</footer>
        )}
      </div>
    </div>
  );
}

/**
 * A button for an action that cannot be undone with a click.
 *
 * Two-step by construction: the first click arms it, the second performs it.
 * HELM can unload the model six agents are using; that should cost two
 * deliberate clicks, not one stray one.
 */
export function ConfirmButton({ label, armedLabel, onConfirm, disabled, title }: {
  label: string; armedLabel?: string;
  onConfirm: () => void; disabled?: boolean; title?: string;
}) {
  const [armed, setArmed] = useState(false);
  if (!armed) {
    return (
      <button type="button" className="btn-secondary" disabled={disabled} title={title}
        onClick={() => setArmed(true)}>
        {label}
      </button>
    );
  }
  return (
    <span style={{ display: 'inline-flex', gap: '0.25rem' }}>
      <button type="button" className="btn-danger" onClick={() => { setArmed(false); onConfirm(); }}>
        {armedLabel ?? `Confirm ${label.toLowerCase()}`}
      </button>
      <button type="button" className="btn-secondary" onClick={() => setArmed(false)}>Cancel</button>
    </span>
  );
}

export function formatBytes(n: number | null): string {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}
