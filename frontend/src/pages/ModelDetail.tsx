/**
 * The dedicated per-model detail page (14c §1.12, the "dedicated view" surface).
 *
 * Same data and same honesty as the dashboard's expandable row — it shares
 * `ModelDetailView` so they cannot diverge — but with room to breathe and its
 * own actions (Settings, Update). Reached from a dashboard row; a Back link
 * returns there. Fetches /detail on mount and on a gentle refresh; NEVER a
 * fast poll, because /detail is three upstream reads.
 */
import { useCallback, useEffect, useState } from 'react';
import { api, type ModelDetail } from '../api/client';
import { ModelDetailView } from '../components/ModelDetailView';

/** Gentle self-refresh so an open detail page doesn't go stale, without
 *  hammering the box with three reads every few seconds. */
const REFRESH_MS = 30000;

export default function ModelDetailPage({ modelId, onBack, onConfigure, onUpdate, managementEnabled }: {
  modelId: string;
  onBack: () => void;
  onConfigure?: (modelName: string) => void;
  onUpdate?: (modelName: string) => void;
  managementEnabled?: boolean;
}) {
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.modelDetail(modelId));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [modelId]);

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <button type="button" className="btn-secondary" onClick={onBack} style={{ alignSelf: 'flex-start' }}>
        ← Back to dashboard
      </button>

      <div className="card">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', flexWrap: 'wrap' }}>
          <h2 style={{ margin: 0, fontSize: '1.25rem' }}>{modelId}</h2>
          {detail?.checkpoint && (
            <span className="stat-sub mono" style={{ wordBreak: 'break-all' }}>{detail.checkpoint}</span>
          )}
        </div>

        {managementEnabled && (
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
            {onConfigure && (
              <button type="button" className="btn-secondary" onClick={() => onConfigure(modelId)}>Settings</button>
            )}
            {onUpdate && (
              <button type="button" className="btn-secondary" onClick={() => onUpdate(modelId)}>Update…</button>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="card">
          <span className="unknown">
            Could not load live detail: {error}
          </span>
          <div className="stat-sub" style={{ marginTop: '0.375rem' }}>
            If the model was just unloaded (or the box cleared its loaded set), it has no live
            detail to read. This is the honest 404, not a HELM failure.
          </div>
        </div>
      )}

      {!detail && !error && <p className="stat-sub">Reading live detail…</p>}

      {detail && (
        <div className="card">
          <ModelDetailView detail={detail} />
        </div>
      )}
    </div>
  );
}
