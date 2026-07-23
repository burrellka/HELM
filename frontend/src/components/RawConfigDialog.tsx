/**
 * Level A raw config editor. Spec §5.3b, 14c §2.3.
 *
 * The LocalAI-style escape hatch: edit a model's raw recipe_options directly,
 * for keys the structured form does not surface. It is more powerful than the
 * form, not more correct, so it runs the SAME guarded pipeline the form's
 * validation does: read current first (never a blank editor), validate on
 * every change, show the diff before apply, back up the prior version, confirm
 * the reload, apply, and keep a one-click restore history.
 *
 * Applying reloads the one model. That is the whole blast radius, and the
 * confirm names it.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  ApiError,
  api,
  type ConfigBackup,
  type ConfigValidation,
  type ModelConfigRead,
} from '../api/client';
import { Callout, Modal } from './common';

export default function RawConfigDialog({ modelId, onClose, onApplied }: {
  modelId: string; onClose: () => void; onApplied: () => void;
}) {
  const [read, setRead] = useState<ModelConfigRead | null>(null);
  const [text, setText] = useState('');
  const [validation, setValidation] = useState<ConfigValidation | null>(null);
  const [backups, setBackups] = useState<ConfigBackup[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scope = `model:${modelId}`;

  const loadBackups = useCallback(() => {
    api.configBackups(scope).then((r) => setBackups(r.backups)).catch(() => {});
  }, [scope]);

  // Read-first: never present a blank editor (14c §2.3 step 1).
  useEffect(() => {
    api.readModelConfig(modelId).then((r) => {
      setRead(r);
      setText(r.running_text);
    }).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    loadBackups();
  }, [modelId, loadBackups]);

  // Validate on change, debounced. The diff and issues come from the server,
  // which runs the same F4 reserved-arg check the structured form uses.
  useEffect(() => {
    if (!read) return;
    const id = setTimeout(() => {
      api.validateModelConfig(modelId, text).then(setValidation).catch(() => setValidation(null));
    }, 300);
    return () => clearTimeout(id);
  }, [text, read, modelId]);

  const apply = async (confirm: boolean) => {
    setBusy(true); setError(null); setMessage(null);
    try {
      const r = await api.applyModelConfig(modelId, text, confirm);
      if (!r.applied) {
        setError(`${r.error_kind}: ${r.error_detail}`);
        return;
      }
      setMessage(
        r.verified
          ? `Applied and verified. Backup #${r.backup_id} saved.`
          : `Applied, but HELM could not confirm the change read back. Backup #${r.backup_id} saved. Check the dashboard.`,
      );
      loadBackups();
      onApplied();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return apply(true);  // confirm reload
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = async (confirm: boolean) => {
    setBusy(true); setError(null); setMessage(null);
    try {
      const r = await api.resetModelConfig(modelId, confirm);
      setMessage(r.cleared
        ? 'Reset to auto-detect. Saved overrides cleared.'
        : (r.note ?? 'Reset reloaded the model.'));
      loadBackups();
      onApplied();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return reset(true);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const restore = async (backupId: number, confirm: boolean) => {
    setBusy(true); setError(null);
    try {
      await api.restoreBackup(backupId, confirm);
      setMessage(`Restored from backup #${backupId}.`);
      onApplied();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return restore(backupId, true);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const blocked = validation?.blocked ?? false;
  const changed = read ? text.trim() !== read.running_text.trim() : false;

  return (
    <Modal
      title={`Raw options — ${modelId}`}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn-secondary" onClick={onClose}>Close</button>
          <button type="button" className="btn-secondary" disabled={busy}
            title="Clear saved overrides and reload with GGUF auto-detect."
            onClick={() => void reset(false)}>Reset to auto</button>
          <button type="button" className="btn-primary" disabled={busy || blocked || !changed}
            title={blocked ? 'Fix the errors first' : !changed ? 'No changes to apply' : undefined}
            onClick={() => void apply(false)}>
            {busy ? 'Working…' : 'Apply (reloads model)'}
          </button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {error && <Callout tone="bad" title="Failed">{error}</Callout>}
        {message && <Callout tone="good">{message}</Callout>}

        <Callout tone="info">
          This edits the model's raw <code className="mono">recipe_options</code> directly, the
          escape hatch for keys the form does not show. It runs the same checks as the form:
          reserved arguments are rejected, and applying reloads the model. The prior version is
          backed up before every write.
        </Callout>

        <textarea
          value={text}
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          rows={10}
          className="mono"
          style={{
            fontSize: '0.8125rem', padding: '0.625rem', borderRadius: 8,
            border: `1px solid ${blocked ? 'var(--status-bad)' : 'var(--border-strong)'}`,
            background: 'var(--bg-subtle)', resize: 'vertical', lineHeight: 1.5,
          }}
        />

        {validation && validation.issues.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
            {validation.issues.map((iss, i) => (
              <Callout key={i} tone={iss.level === 'error' ? 'bad'
                : iss.level === 'warning' ? 'warn' : 'info'}>
                {iss.key && <code className="mono">{iss.key}</code>}{iss.key && ': '}{iss.message}
              </Callout>
            ))}
          </div>
        )}

        {validation && validation.changes.length > 0 && (
          <div>
            <div className="stat-sub" style={{ marginBottom: '0.25rem' }}>Changes before apply:</div>
            <table style={{ fontSize: '0.8125rem', borderCollapse: 'collapse' }}>
              <tbody>
                {validation.changes.map((c) => (
                  <tr key={c.key}>
                    <td className="mono" style={{ padding: '0.125rem 0.75rem 0.125rem 0' }}>{c.key}</td>
                    <td className="mono stat-sub" style={{ padding: '0.125rem 0.75rem 0.125rem 0' }}>
                      {c.kind === 'added' ? '(new)' : JSON.stringify(c.old)}
                    </td>
                    <td className="mono" style={{ padding: '0.125rem 0' }}>
                      {c.kind === 'removed' ? '(removed)' : JSON.stringify(c.new)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {backups.length > 0 && (
          <details>
            <summary style={{ cursor: 'pointer', fontSize: '0.8125rem', fontWeight: 600 }}>
              Restore history ({backups.length})
            </summary>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', marginTop: '0.5rem' }}>
              {backups.map((b) => (
                <div key={b.id} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.8125rem' }}>
                  <button type="button" className="btn-secondary" disabled={busy}
                    onClick={() => void restore(b.id, false)}>Restore</button>
                  <span className="stat-sub">{b.reason} · {new Date(b.created_at).toLocaleString()}</span>
                  <code className="mono stat-sub" style={{
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 260,
                  }}>{b.preview}</code>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </Modal>
  );
}
