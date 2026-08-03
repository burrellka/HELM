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
  type HFHit,
  type LibraryEntry,
  type LibraryQuery,
  type Library as LibraryData,
} from '../api/client';
import { Badge, Callout, ConfirmButton, Modal, Unknown, formatBytes } from '../components/common';
import LoadDialog, { type LoadPrefill } from '../components/LoadDialog';
import RawConfigDialog from '../components/RawConfigDialog';
import AnalyzePanel from '../components/AnalyzePanel';
import Markdown from '../components/Markdown';

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
  const [pulling, setPulling] = useState<{ entry: LibraryEntry; mode: 'download' | 'update' } | null>(null);
  /** Which side of the drift the editor opens on: what's running, or what's saved. */
  const [prefillSide, setPrefillSide] = useState<'running' | 'saved'>('running');
  /** Live HuggingFace discovery for the current search term (null = not searched). */
  const [hf, setHf] = useState<{ results: HFHit[]; error: string | null } | null>(null);
  const [hfBusy, setHfBusy] = useState(false);

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

  // Live HuggingFace search alongside the catalog (Lemonade's UI does this; our
  // catalog-only search was the "0 results" surprise). Its own slower debounce —
  // it's an external call — and only for real search terms.
  useEffect(() => {
    const text = (query.text ?? '').trim();
    if (text.length < 2) { setHf(null); setHfBusy(false); return; }
    setHfBusy(true);
    let live = true;
    const id = setTimeout(async () => {
      try {
        const r = await api.hfSearch(text);
        if (live) setHf({ results: r.results, error: r.error });
      } catch (e) {
        if (live) setHf({ results: [], error: e instanceof Error ? e.message : String(e) });
      } finally {
        if (live) setHfBusy(false);
      }
    }, 450);
    return () => { live = false; clearTimeout(id); };
  }, [query.text]);

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
            onPull={(mode) => setPulling({ entry: e, mode })}
            onChanged={refresh}
          />
        ))}
        {data?.entries.length === 0 && (
          <div className="card"><span className="stat-sub">
            No models in your catalog match{(query.text ?? '').trim() ? ` "${query.text}"` : ''}.
          </span></div>
        )}
      </div>

      {(query.text ?? '').trim().length >= 2 && (
        <HuggingFaceResults
          hf={hf}
          busy={hfBusy}
          managementEnabled={managementEnabled}
          catalogCheckpoints={new Set(
            (data?.entries ?? [])
              .map((e) => (e.checkpoint ?? '').split(':')[0])
              .filter(Boolean),
          )}
          onDownloaded={refresh}
        />
      )}

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
      {pulling && (
        <PullDialog
          entry={pulling.entry}
          mode={pulling.mode}
          onClose={() => setPulling(null)}
          onStarted={() => { setPulling(null); void refresh(); }}
        />
      )}
    </div>
  );
}

/**
 * Live HuggingFace discovery, shown under the catalog results — the capability
 * Lemonade's own search has and HELM's catalog-only search didn't (the "0 of
 * 162" surprise). A failed HF call is a visible "couldn't search", never a
 * silent empty list that reads as "nothing exists".
 */
function HuggingFaceResults({ hf, busy, managementEnabled, catalogCheckpoints, onDownloaded }: {
  hf: { results: HFHit[]; error: string | null } | null;
  busy: boolean;
  managementEnabled: boolean;
  catalogCheckpoints: Set<string>;
  onDownloaded: () => void;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div className="stat-sub" style={{
        textTransform: 'uppercase', letterSpacing: '0.02em', fontWeight: 600, marginTop: '0.5rem',
      }}>
        From HuggingFace{busy && <span style={{ fontWeight: 400 }}> · searching…</span>}
      </div>
      {hf?.error && (
        <Callout tone="warn" title="Couldn't search HuggingFace">
          {hf.error} — this doesn't affect your catalog results above.
        </Callout>
      )}
      {hf && !hf.error && hf.results.length === 0 && !busy && (
        <div className="card"><span className="stat-sub">No GGUF models on HuggingFace match.</span></div>
      )}
      {hf?.results.map((h) => (
        <HFResultRow key={h.id} hit={h} managementEnabled={managementEnabled}
          alreadyInCatalog={catalogCheckpoints.has(h.id)} onDownloaded={onDownloaded} />
      ))}
    </div>
  );
}

function HFResultRow({ hit, managementEnabled, alreadyInCatalog, onDownloaded }: {
  hit: HFHit; managementEnabled: boolean; alreadyInCatalog: boolean; onDownloaded: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const download = async () => {
    setBusy(true); setNote(null);
    try {
      // Pull under a user.* name with the HF repo as checkpoint — the custom-model
      // path Lemonade's /pull already supports. Surface whatever it says back
      // (started, warnings, or an error) rather than assuming success.
      const r = await api.pull(hit.suggested_name, hit.id);
      setNote(`Download started as ${hit.suggested_name}. Watch the header indicator.`
        + (r.warnings?.length ? ' ' + r.warnings.join(' ') : ''));
      onDownloaded();
    } catch (e) {
      setNote(`Could not start the download: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{
      display: 'flex', alignItems: 'center', gap: '1rem',
      justifyContent: 'space-between', flexWrap: 'wrap',
    }}>
      <div style={{ minWidth: 0 }}>
        <a href={hit.url} target="_blank" rel="noreferrer" style={{ fontWeight: 600 }}>{hit.id}</a>
        <div className="stat-sub" style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {hit.downloads != null && <span>↓ {hit.downloads.toLocaleString()}</span>}
          {hit.likes != null && hit.likes > 0 && <span>♥ {hit.likes}</span>}
          <span>GGUF</span>
        </div>
        {note && <div className="stat-sub" style={{ marginTop: '0.25rem' }}>{note}</div>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
        {alreadyInCatalog
          ? <span className="stat-sub">already in your library</span>
          : managementEnabled
            ? <button type="button" className="btn-secondary" disabled={busy}
                onClick={() => void download()}>{busy ? 'Starting…' : 'Download'}</button>
            : <a href={hit.url} target="_blank" rel="noreferrer" className="btn-secondary">View on HF</a>}
      </div>
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

function EntryCard({ entry, managementEnabled, onEdit, onLoad, onRevert, onRawConfig, onAnalyze, onPull, onChanged }: {
  entry: LibraryEntry; managementEnabled: boolean;
  onEdit: () => void; onLoad: () => void;
  onRevert: (entry: LibraryEntry) => void; onRawConfig: () => void;
  onAnalyze: () => void; onPull: (mode: 'download' | 'update') => void; onChanged: () => void;
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

          <ModelCardSection entry={entry} />

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
          {managementEnabled && entry.downloaded && entry.update_available && (
            <button type="button" className="btn-primary" onClick={() => onPull('update')}
              title="Lemonade found a newer version of this model on its HuggingFace source. Review and re-download.">
              Update
            </button>
          )}
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
            <button type="button" className="btn-primary" onClick={() => onPull('download')}
              title="See the model card and details, then download.">Download</button>
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

/**
 * The download / update preview. The operator should never pull gigabytes blind:
 * this shows the model's HuggingFace source (repo, when it last changed,
 * popularity, license — the cached metadata the advisor uses) AND the actual
 * model card (the README), so they can read what they're getting first. Then it
 * pulls. Progress shows in the header download indicator.
 *
 * `update` mode adds the "newer version available" framing and pulls by name
 * (a built-in updates by re-pulling its own name — there is no separate Lemonade
 * update call). `download` mode passes the catalog checkpoint/recipe for a
 * not-yet-downloaded model (the backend keeps them only for user. custom pulls).
 */
function PullDialog({ entry, mode, onClose, onStarted }: {
  entry: LibraryEntry; mode: 'download' | 'update';
  onClose: () => void; onStarted: () => void;
}) {
  const [external, setExternal] = useState<Record<string, any> | null>(null);
  const [card, setCard] = useState<{ card: string | null; note: string | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Cached HF metadata (source, popularity) + the model card (README). Both
    // are honest about absence for a custom/local model.
    api.analyze(entry.model_id, false).then((r) => setExternal(r.external)).catch(() => setExternal(null));
    api.modelCard(entry.model_id).then(setCard).catch(() => setCard({ card: null, note: 'Could not load the model card.' }));
  }, [entry.model_id]);

  const doPull = async () => {
    setBusy(true); setError(null);
    try {
      if (mode === 'update') {
        await api.pull(entry.model_id);   // built-in re-pull by name → latest
      } else {
        await api.pull(entry.model_id, entry.checkpoint ?? undefined, entry.recipe ?? undefined);
      }
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const meta = external?.status === 'found' ? external.metadata : null;
  const lastModified = meta?.last_modified ? new Date(meta.last_modified).toLocaleDateString() : null;
  const verb = mode === 'update' ? 'Download the update' : 'Download';

  return (
    <Modal
      title={`${mode === 'update' ? 'Update' : 'Download'} — ${entry.model_id}`}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button type="button" className="btn-primary" disabled={busy} onClick={() => void doPull()}>
            {busy ? 'Starting…' : verb}
          </button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
        {mode === 'update' ? (
          <Callout tone="warn" title="A newer version is available">
            Lemonade found that this model's source has been updated since your copy was downloaded.
            Updating re-downloads the latest version{entry.size ? ` (about ${formatBytes(entry.size)})` : ''}.
          </Callout>
        ) : (
          <Callout tone="info" title="Review before downloading">
            The model card below is from HuggingFace. Read what this model is and how it's meant to be
            run before pulling it{entry.size ? ` (about ${formatBytes(entry.size)})` : ''}.
          </Callout>
        )}

        <div>
          <div className="stat-sub">Source</div>
          <div className="mono" style={{ fontSize: '0.8125rem', wordBreak: 'break-all' }}>
            {entry.checkpoint || external?.repo_id || 'unknown'}
          </div>
        </div>

        {meta && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
            {lastModified && <Fact label="Source updated" value={lastModified} />}
            {meta.downloads != null && <Fact label="Downloads" value={meta.downloads.toLocaleString()} />}
            {meta.likes != null && <Fact label="Likes" value={meta.likes} />}
            {meta.license && <Fact label="License" value={meta.license} />}
          </div>
        )}

        {/* The model card. The one thing that tells the operator what they're
            getting. Scrolls in its own box so a long card doesn't run away. */}
        <div>
          <div className="stat-sub" style={{ marginBottom: '0.375rem' }}>Model card (HuggingFace README)</div>
          {card === null ? (
            <p className="stat-sub" style={{ margin: 0 }}>Loading the model card…</p>
          ) : card.card ? (
            <div style={{
              maxHeight: '42vh', overflowY: 'auto', padding: '0.75rem 0.875rem',
              background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', borderRadius: 8,
            }}>
              <Markdown>{card.card}</Markdown>
            </div>
          ) : (
            <Callout tone="info">{card.note ?? 'No model card available for this model.'}</Callout>
          )}
        </div>

        <p className="stat-sub" style={{ margin: 0 }}>
          The download runs on the server and shows in the indicator at the top.
          {mode === 'update' && entry.loaded
            ? ' This model is loaded — the running copy keeps serving the old version until you reload it (Settings → apply, or unload and load) after the download finishes.'
            : ''}
        </p>

        {error && <Callout tone="bad" title={`${mode === 'update' ? 'Update' : 'Download'} could not start`}>{error}</Callout>}
      </div>
    </Modal>
  );
}

/** A small labelled fact, matching the Analyze panel's style. */
function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="stat-sub">{label}</div>
      <div style={{ fontSize: '0.9375rem', fontWeight: 600 }}>{value}</div>
    </div>
  );
}

/** Derive a HuggingFace repo from a checkpoint client-side, or null. A checkpoint
 *  like `org/repo:quant` is an HF repo; a local file path (starts with '/' or has
 *  a backslash) is not. */
function hfRepoFromCheckpoint(checkpoint: string | null): string | null {
  if (!checkpoint || checkpoint.startsWith('/') || checkpoint.includes('\\')) return null;
  const repo = checkpoint.split(':')[0];
  return /^[^/\s]+\/[^/\s]+$/.test(repo) ? repo : null;
}

/**
 * Per-row HuggingFace link + a collapsed-by-default description (the model card).
 * The link is instant (derived from the checkpoint); the card is lazy — fetched
 * only when the operator expands it, so browsing the library never fans out a
 * pile of HF calls. Renders nothing for a model with no resolvable HF repo.
 */
function ModelCardSection({ entry }: { entry: LibraryEntry }) {
  const [open, setOpen] = useState(false);
  const [card, setCard] = useState<{ card: string | null; note: string | null } | null>(null);
  const [loading, setLoading] = useState(false);
  const [repoId, setRepoId] = useState<string | null>(hfRepoFromCheckpoint(entry.checkpoint));

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (next && card === null && !loading) {
      setLoading(true);
      try {
        const r = await api.modelCard(entry.model_id);
        setCard({ card: r.card, note: r.note });
        if (r.repo_id) setRepoId(r.repo_id);   // covers a remembered-checkpoint repo
      } catch {
        setCard({ card: null, note: 'Could not load the model card.' });
      } finally {
        setLoading(false);
      }
    }
  };

  if (!repoId) return null;   // no HF page and no card to point at

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <a href={`https://huggingface.co/${repoId}`} target="_blank" rel="noreferrer noopener"
        style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--accent-primary)' }}>
        View on HuggingFace ↗
      </a>
      <div style={{ marginTop: '0.25rem' }}>
        <button type="button" onClick={() => void toggle()}
          style={{
            background: 'none', border: 'none', cursor: 'pointer', padding: 0,
            color: 'var(--text-muted)', fontSize: '0.8125rem', fontWeight: 600,
            display: 'flex', alignItems: 'center', gap: '0.3rem',
          }}>
          <span>{open ? '▾' : '▸'}</span> Description
        </button>
        {open && (
          <div style={{
            marginTop: '0.375rem', maxHeight: '34vh', overflowY: 'auto',
            padding: '0.625rem 0.75rem', background: 'var(--bg-subtle)',
            border: '1px solid var(--border-color)', borderRadius: 8,
          }}>
            {loading ? (
              <p className="stat-sub" style={{ margin: 0 }}>Loading the model card…</p>
            ) : card?.card ? (
              <Markdown>{card.card}</Markdown>
            ) : (
              <span className="stat-sub">{card?.note ?? 'No model card available.'}</span>
            )}
          </div>
        )}
      </div>
    </div>
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
