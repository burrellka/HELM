/**
 * Phase 3: the model library. Spec §5.2, §5.3a.
 *
 * **The capability badges are the point, and Rule 10 is why.** On the NPU, tool
 * calling is compiled per-model into the kernel; FLM silently strips the
 * `tools` array for models that lack it, and the model hallucinates instead of
 * calling anything. The `tool-calling` label is the ONLY reliable indicator —
 * not the name, not the family, not the size.
 *
 * So this page renders three distinct states and never conflates them:
 *   - `matched`   → badges, as published.
 *   - `no_labels` → "no capability labels", plus a hard warning for FLM
 *                   models: an agent wired to this WILL fail silently.
 *   - `unknown`   → "capabilities unknown". The join failed. We know nothing,
 *                   and an empty badge row would look exactly like a model
 *                   that genuinely lacks the capability (F9).
 *
 * All filtering is server-side: `domain.library.search` is where the §5.3a
 * query semantics are tested, and this page sends intent rather than
 * predicates.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  api,
  type Drift,
  type LibraryEntry,
  type LibraryQuery,
  type Library as LibraryData,
} from '../api/client';
import { Badge, Callout, ConfirmButton, Modal, Unknown, formatBytes } from '../components/common';
import LoadDialog, { type LoadPrefill } from '../components/LoadDialog';
import RawConfigDialog from '../components/RawConfigDialog';
import AnalyzePanel from '../components/AnalyzePanel';

/**
 * Prefill the editor from a side the SERVER already derived.
 *
 * `'running'` opens on what the model is doing now — the right default,
 * because that is the thing being changed. `'saved'` is the Revert path:
 * reload using the persisted options and discard the running ones.
 *
 * This deliberately does no arithmetic and no arg parsing. The first version
 * of it divided ctx_size by slots and stripped --parallel with a regex, which
 * is a second implementation of Rule 1 and a second arg parser — exactly the
 * pair F4 says will drift (that regex already missed `-np`). Both now come off
 * `saved_settings` / `running_settings`, computed by the same tested code that
 * powers the dashboard's Concurrency column.
 */
function prefillFrom(entry: LibraryEntry, which: 'running' | 'saved'): LoadPrefill {
  const s = which === 'running'
    ? (entry.running_settings ?? entry.saved_settings)
    : (entry.saved_settings ?? entry.running_settings);
  return {
    slots: s?.slots ?? null,
    ctxPerSlot: s?.ctx_per_slot ?? null,
    backend: s?.backend ?? null,
    temperature: s?.temperature ?? null,
    topK: s?.top_k ?? null,
    topP: s?.top_p ?? null,
    minP: s?.min_p ?? null,
    repeatPenalty: s?.repeat_penalty ?? null,
    extraArgs: s?.extra_args ?? '',
  };
}

export default function Library({ managementEnabled, focusModelId }: {
  managementEnabled: boolean;
  /** Set when the operator arrived from the dashboard's Settings button, so
   *  they land on the one model they clicked rather than 162 of them. */
  focusModelId?: string | null;
}) {
  const [data, setData] = useState<LibraryData | null>(null);
  const [query, setQuery] = useState<LibraryQuery>({ text: focusModelId ?? '' });
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<LibraryEntry | null>(null);
  const [loading, setLoading] = useState<LibraryEntry | null>(null);
  const [rawConfig, setRawConfig] = useState<LibraryEntry | null>(null);
  const [analyzing, setAnalyzing] = useState<LibraryEntry | null>(null);
  /** Which side of the drift the editor opens on: what's running, or what's saved. */
  const [prefillSide, setPrefillSide] = useState<'running' | 'saved'>('running');

  const refresh = useCallback(async () => {
    try {
      setData(await api.library(query));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [query]);

  useEffect(() => {
    const id = setTimeout(() => void refresh(), 200);  // debounce typing
    return () => clearTimeout(id);
  }, [refresh]);

  const set = (patch: Partial<LibraryQuery>) => setQuery((q) => ({ ...q, ...patch }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {error && <Callout tone="bad" title="Could not read the library">{error}</Callout>}

      {/* F9 in the wild: loaded models with no catalog row. HELM's badges are
          blind for these, and the operator should know which rows it can't
          vouch for rather than reading a blank as "no capabilities". */}
      {data && data.unmatched_loaded.length > 0 && (
        <Callout tone="warn" title="Some loaded models are not in the catalog">
          {data.unmatched_loaded.join(', ')} — HELM cannot look up capabilities for these, so
          their badges read "unknown" rather than empty. Treat any tool-calling assumption about
          them as unverified.
        </Callout>
      )}

      <Filters query={query} set={set} knownTags={data?.known_tags ?? []} data={data} />

      {data && (
        <div className="stat-sub">
          Showing {data.entries.length} of {data.total} models.
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {data?.entries.map((e) => (
          <EntryCard
            key={e.model_id}
            entry={e}
            managementEnabled={managementEnabled}
            onEdit={() => setEditing(e)}
            onLoad={() => { setPrefillSide('running'); setLoading(e); }}
            onRevert={(entry) => { setPrefillSide('saved'); setLoading(entry); }}
            onRawConfig={() => setRawConfig(e)}
            onAnalyze={() => setAnalyzing(e)}
            onChanged={refresh}
          />
        ))}
        {data?.entries.length === 0 && (
          <div className="card"><span className="stat-sub">No models match.</span></div>
        )}
      </div>

      {editing && (
        <MetadataDialog entry={editing} onClose={() => setEditing(null)}
          onSaved={() => { setEditing(null); void refresh(); }} />
      )}
      {loading && (
        <LoadDialog
          entry={loading}
          mode={loading.loaded ? 'reconfigure' : 'load'}
          prefill={prefillFrom(loading, prefillSide)}
          onClose={() => setLoading(null)}
          onDone={() => { setLoading(null); void refresh(); }}
        />
      )}
      {rawConfig && (
        <RawConfigDialog
          modelId={rawConfig.model_id}
          onClose={() => setRawConfig(null)}
          onApplied={() => void refresh()}
        />
      )}
      {analyzing && (
        <Modal title={`Analyze — ${analyzing.model_id}`} onClose={() => setAnalyzing(null)}>
          <AnalyzePanel modelId={analyzing.model_id} />
        </Modal>
      )}
    </div>
  );
}

function Filters({ query, set, knownTags, data }: {
  query: LibraryQuery;
  set: (patch: Partial<LibraryQuery>) => void;
  knownTags: string[];
  data: LibraryData | null;
}) {
  const toggle = (key: keyof LibraryQuery) => () =>
    set({ [key]: query[key] ? undefined : true } as Partial<LibraryQuery>);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
      <input
        value={query.text ?? ''}
        placeholder="Search name, checkpoint, description, tags…"
        onChange={(e) => set({ text: e.target.value })}
        style={{
          fontFamily: 'inherit', fontSize: '0.9375rem', padding: '0.5rem 0.625rem',
          border: '1px solid var(--border-strong)', borderRadius: 8,
        }}
      />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', alignItems: 'center' }}>
        <Chip on={!!query.downloaded} onClick={toggle('downloaded')}>Downloaded</Chip>
        <Chip on={!!query.loaded} onClick={toggle('loaded')}>Loaded</Chip>
        <Chip on={!!query.favorite} onClick={toggle('favorite')}>★ Favourites</Chip>
        <Chip on={!!query.include_archived} onClick={toggle('include_archived')}>Show archived</Chip>
        <span style={{ width: 1, height: 20, background: 'var(--border-color)' }} />
        {['tool-calling', 'vision', 'reasoning', 'mtp', 'embeddings'].map((label) => (
          <Chip
            key={label}
            on={(query.labels ?? '').split(',').includes(label)}
            onClick={() => {
              const current = new Set((query.labels ?? '').split(',').filter(Boolean));
              current.has(label) ? current.delete(label) : current.add(label);
              set({ labels: [...current].join(',') });
            }}
          >{label}</Chip>
        ))}
        <span style={{ width: 1, height: 20, background: 'var(--border-color)' }} />
        {['llamacpp', 'flm'].map((recipe) => (
          <Chip
            key={recipe}
            on={(query.recipes ?? '').split(',').includes(recipe)}
            onClick={() => {
              const current = new Set((query.recipes ?? '').split(',').filter(Boolean));
              current.has(recipe) ? current.delete(recipe) : current.add(recipe);
              set({ recipes: [...current].join(',') });
            }}
          >{recipe === 'flm' ? 'flm (NPU)' : 'llamacpp (GPU)'}</Chip>
        ))}
      </div>
      {knownTags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', alignItems: 'center' }}>
          <span className="stat-sub">Your tags:</span>
          {knownTags.map((tag) => (
            <Chip
              key={tag}
              on={(query.tags ?? '').split(',').includes(tag)}
              onClick={() => {
                const current = new Set((query.tags ?? '').split(',').filter(Boolean));
                current.has(tag) ? current.delete(tag) : current.add(tag);
                set({ tags: [...current].join(',') });
              }}
            >{tag}</Chip>
          ))}
        </div>
      )}
      {data && (
        <a
          className="stat-sub"
          href="/api/v1/library/export"
          style={{ color: 'var(--accent-primary)', textDecoration: 'none' }}
        >
          Export your descriptions, tags and favourites (JSON) — this is the only copy.
        </a>
      )}
    </div>
  );
}

function Chip({ on, onClick, children }: {
  on: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        fontFamily: 'inherit', fontSize: '0.75rem', fontWeight: 500, cursor: 'pointer',
        padding: '0.25rem 0.625rem', borderRadius: 999,
        border: `1px solid ${on ? 'var(--accent-primary)' : 'var(--border-strong)'}`,
        background: on ? 'var(--accent-soft)' : 'var(--bg-secondary)',
        color: on ? 'var(--accent-text)' : 'var(--text-secondary)',
      }}
    >{children}</button>
  );
}

function EntryCard({ entry, managementEnabled, onEdit, onLoad, onRevert, onRawConfig, onAnalyze, onChanged }: {
  entry: LibraryEntry; managementEnabled: boolean;
  onEdit: () => void; onLoad: () => void;
  onRevert: (entry: LibraryEntry) => void; onRawConfig: () => void;
  onAnalyze: () => void; onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);

  const toggleFavorite = async () => {
    setBusy(true);
    try {
      await api.updateMetadata(entry.model_id, { favorite: !entry.favorite });
      onChanged();
    } finally { setBusy(false); }
  };

  return (
    <div className="card" style={{ opacity: entry.archived ? 0.6 : 1 }}>
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
        <button
          type="button" onClick={toggleFavorite} disabled={busy}
          title={entry.favorite ? 'Unfavourite' : 'Favourite'}
          style={{
            background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.125rem',
            color: entry.favorite ? 'var(--status-warn)' : 'var(--text-muted)', lineHeight: 1,
          }}
        >{entry.favorite ? '★' : '☆'}</button>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <strong>{entry.model_id}</strong>
            {entry.loaded && <Badge tone="good">loaded</Badge>}
            {entry.pinned && <Badge tone="accent">pinned</Badge>}
            {entry.update_available && <Badge tone="warn">update available</Badge>}
            {entry.suggested && <Badge tone="neutral" title="Lemonade marks this recommended for this hardware">suggested</Badge>}
            {entry.archived && <Badge tone="neutral">archived</Badge>}
          </div>

          <div className="stat-sub mono" style={{
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>{entry.checkpoint}</div>

          {entry.description && (
            <div style={{ fontSize: '0.875rem', marginTop: '0.375rem' }}>{entry.description}</div>
          )}

          <Capabilities entry={entry} />

          {entry.tags.length > 0 && (
            <div style={{ display: 'flex', gap: '0.25rem', marginTop: '0.375rem', flexWrap: 'wrap' }}>
              {entry.tags.map((t) => <Badge key={t} tone="neutral">{t}</Badge>)}
            </div>
          )}

          {entry.drift && (
            <SettingsPanel
              drift={entry.drift}
              canWrite={managementEnabled}
              onRevert={() => onRevert(entry)}
            />
          )}

          <div className="stat-sub" style={{ marginTop: '0.375rem' }}>
            {entry.recipe} · {entry.downloaded ? formatBytes(entry.size) : 'not downloaded'}
            {entry.max_context_window ? ` · max ctx ${entry.max_context_window.toLocaleString()}` : ''}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', alignItems: 'flex-end' }}>
          <button type="button" className="btn-secondary" onClick={onAnalyze}
            title="What is this model: quant, fit, capabilities, and a grounded summary.">
            Analyze
          </button>
          <button type="button" className="btn-secondary" onClick={onEdit}>Notes &amp; tags</button>
          {/* The §5.3 editor. Available whether or not the model is resident:
              for a loaded model it reloads with the new options, for an
              unloaded one it is the load dialog. Same fields either way. */}
          {managementEnabled && entry.downloaded && entry.loaded && (
            <button type="button" className="btn-secondary" onClick={onLoad}
              title="View and change slots, context, backend and sampling. Applying reloads the model.">
              Settings
            </button>
          )}
          {managementEnabled && entry.downloaded && entry.loaded && (
            <button type="button" className="btn-secondary" onClick={onRawConfig}
              title="Edit the raw recipe_options JSON directly. The power-user escape hatch, same guardrails.">
              Raw
            </button>
          )}
          {managementEnabled && entry.downloaded && !entry.loaded && (
            <button type="button" className="btn-primary" onClick={onLoad}>Load</button>
          )}
          {managementEnabled && entry.downloaded && !entry.loaded && (
            <ConfirmButton
              label="Delete"
              armedLabel="Delete weights"
              title="Removes the weights from disk. Re-downloading may be many GB."
              onConfirm={async () => { await api.deleteModel(entry.model_id, true); onChanged(); }}
            />
          )}
          {!entry.downloaded && managementEnabled && (
            <button
              type="button" className="btn-secondary"
              onClick={async () => { await api.pull(entry.model_id, entry.checkpoint ?? undefined,
                entry.recipe ?? undefined); onChanged(); }}
            >Download</button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Rule 10, rendered.
 *
 * The three states are visually distinct on purpose. "No labels" and "unknown"
 * would be the same blank space in a naive implementation, and they mean
 * opposite things: one is a finding, the other is an absence of one.
 */
function Capabilities({ entry }: { entry: LibraryEntry }) {
  if (entry.label_join === 'unknown') {
    return (
      <div style={{ marginTop: '0.375rem' }}>
        <Unknown>capabilities unknown — HELM could not match this model to the catalog</Unknown>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '0.25rem', marginTop: '0.375rem', flexWrap: 'wrap', alignItems: 'center' }}>
      {entry.labels.map((label) => (
        <Badge
          key={label}
          tone={label === 'tool-calling' ? 'good' : 'accent'}
          title={label === 'tool-calling'
            ? 'This model can actually use tools. On the NPU this is the only reliable indicator.'
            : undefined}
        >{label}</Badge>
      ))}
      {entry.labels.length === 0 && <Unknown>no capability labels</Unknown>}

      {/* The warning that stops an agent being wired to silent failure. */}
      {entry.strips_tools_silently && (
        <span
          title={
            'FastFlowLM compiles tool calling into the model kernel. This model has no ' +
            'tool-calling label, so FLM will SILENTLY DROP the tools array — the model never ' +
            'sees the tools and hallucinates an answer instead. There is no error.'
          }
          style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--status-bad)' }}
        >
          ⚠ will silently strip tools
        </span>
      )}
    </div>
  );
}

/**
 * §5.3's settings panel: SAVED vs RUNNING, side by side, ALWAYS.
 *
 * It used to render only when the two disagreed, which meant a correctly
 * configured model showed nothing at all — you could see your settings only
 * when they were wrong. That is backwards. The panel is now always available
 * for a loaded model; it just opens itself when there is something to worry
 * about.
 *
 * The summary line is the whole feature in one row: nobody else shows you that
 * the 26B is running sampling settings a restart will silently discard.
 */
function SettingsPanel({ drift, onRevert, canWrite }: {
  drift: Drift; onRevert?: () => void; canWrite: boolean;
}) {
  const notable = drift.fields.filter((f) => f.kind !== 'same' && !f.benign);
  const alarming = drift.has_drift || drift.has_unpersisted;
  // Show every field, not just the differing ones: this is the settings view,
  // and "slots: 8, and that IS what's saved" is information too.
  const rows = drift.fields.filter((f) => !(f.kind === 'same' && f.saved == null));

  return (
    <details open={alarming} style={{ marginTop: '0.5rem' }}>
      <summary style={{
        cursor: 'pointer', fontSize: '0.8125rem', fontWeight: 600,
        color: alarming ? 'var(--status-warn)' : 'var(--text-secondary)',
      }}>
        {drift.has_drift
          ? '⚠ Settings — running configuration does not match what is saved'
          : drift.has_unpersisted
            ? '⚠ Settings — running with options that are not saved'
            : 'Settings — saved and running agree'}
      </summary>

      <table style={{ fontSize: '0.8125rem', marginTop: '0.5rem', borderCollapse: 'collapse' }}>
        <thead>
          <tr className="stat-sub" style={{ textAlign: 'left' }}>
            <th style={{ padding: '0.25rem 1rem 0.25rem 0' }}>Field</th>
            <th style={{ padding: '0.25rem 1rem 0.25rem 0' }}>Saved</th>
            <th style={{ padding: '0.25rem 1rem 0.25rem 0' }}>Running now</th>
            <th style={{ padding: '0.25rem 0' }} />
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => {
            const differs = f.kind !== 'same' && !f.benign;
            return (
              <tr key={f.field} title={f.note ?? undefined}>
                <td className="mono" style={{ padding: '0.25rem 1rem 0.25rem 0' }}>{f.field}</td>
                <td className="mono" style={{ padding: '0.25rem 1rem 0.25rem 0' }}>
                  {f.saved == null ? <Unknown>not saved</Unknown> : String(f.saved)}
                </td>
                <td className="mono" style={{
                  padding: '0.25rem 1rem 0.25rem 0',
                  color: differs ? 'var(--status-warn)' : undefined,
                  fontWeight: differs ? 600 : undefined,
                }}>
                  {f.running == null ? <Unknown>—</Unknown> : String(f.running)}
                </td>
                <td className="stat-sub" style={{ padding: '0.25rem 0' }}>
                  {f.kind === 'same' ? '✓' : f.benign ? 'auto-detected' : '⚠ differs'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {notable.length > 0 && (
        <p className="stat-sub" style={{ marginTop: '0.5rem', maxWidth: 640, lineHeight: 1.5 }}>
          A lemond restart reverts the running model to whatever is saved, so anything marked
          "not saved" is lost on the next restart with no warning. Use Settings → tick "save as
          default" to persist what is running, or Revert to put the saved options back.
        </p>
      )}

      {canWrite && onRevert && drift.has_drift && (
        <button type="button" className="btn-secondary" style={{ marginTop: '0.5rem' }}
          onClick={onRevert}
          title="Reload the model using its SAVED options, discarding the running ones.">
          Revert to saved
        </button>
      )}
    </details>
  );
}

function MetadataDialog({ entry, onClose, onSaved }: {
  entry: LibraryEntry; onClose: () => void; onSaved: () => void;
}) {
  const [description, setDescription] = useState(entry.description);
  const [tags, setTags] = useState(entry.tags.join(', '));
  const [archived, setArchived] = useState(entry.archived);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    try {
      await api.updateMetadata(entry.model_id, {
        description,
        archived,
        tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      });
      onSaved();
    } finally { setBusy(false); }
  };

  return (
    <Modal
      title={`Notes & tags — ${entry.model_id}`}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button type="button" className="btn-primary" disabled={busy} onClick={save}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
        <Callout tone="info">
          This is yours, not Lemonade's. It lives in HELM's database, keyed to the model id, and
          it survives restarts and reloads. Nothing upstream knows about it — which is also why
          it is the one thing here that cannot be rebuilt by re-reading the box. Export it.
        </Callout>
        <label className="field">
          <span>Description</span>
          <input value={description} onChange={(e) => setDescription(e.target.value)}
            placeholder="Karl's travel brain · tested-good on multi-step lookups" />
        </label>
        <label className="field">
          <span>Tags (comma separated)</span>
          <input value={tags} onChange={(e) => setTags(e.target.value)}
            placeholder="hermes-preferred, agent-workhorse, uncensored-do-not-deploy" />
          <span className="stat-sub">
            Your own axes, not Lemonade's labels. Tagging the uncensored build
            <code className="mono"> uncensored-do-not-deploy</code> is what stops it being picked
            by accident from a 161-model list.
          </span>
        </label>
        <label style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', fontSize: '0.875rem' }}>
          <input type="checkbox" checked={archived} onChange={(e) => setArchived(e.target.checked)} />
          Archived — hide from the default list without deleting the weights
        </label>
      </div>
    </Modal>
  );
}
