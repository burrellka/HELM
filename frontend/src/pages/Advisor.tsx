/**
 * The Model Advisor tab. Spec 14c Part 1, §1.4, §1.7.
 *
 * Not a raw chat with a model: a chat ABOUT a selected model, pinned to that
 * model's facts. The left column is the Analyze panel (the grounded facts); the
 * right is a conversation anchored to them, and the settings that govern it —
 * the analysis-model picker (default 26B, never hardcoded, sub-2B flagged) and
 * the editable persona. The operational "do not fabricate" layer is locked in
 * the backend and has no field here, by design.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type AdvisorSettings, type LibraryEntry } from '../api/client';
import { Callout } from '../components/common';
import AnalyzePanel from '../components/AnalyzePanel';
import Markdown from '../components/Markdown';

export default function Advisor() {
  const [models, setModels] = useState<LibraryEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    api.library({ downloaded: true }).then((lib) => {
      setModels(lib.entries);
      setSelected((s) => s ?? lib.entries[0]?.model_id ?? null);
    }).catch(() => {});
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <label className="stat-sub">Analyze:</label>
        <select value={selected ?? ''} onChange={(e) => setSelected(e.target.value)}
          style={{ fontFamily: 'inherit', fontSize: '0.9375rem', padding: '0.375rem 0.5rem',
            border: '1px solid var(--border-strong)', borderRadius: 8, minWidth: 260 }}>
          {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id}</option>)}
        </select>
        <button type="button" className="btn-secondary" style={{ marginLeft: 'auto' }}
          onClick={() => setShowSettings((s) => !s)}>
          {showSettings ? 'Hide settings' : 'Advisor settings'}
        </button>
      </div>

      {showSettings && <AdvisorSettingsPanel />}

      {selected && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1.25rem' }}>
          <div className="card"><AnalyzePanel key={selected} modelId={selected} /></div>
          <div className="card"><AdvisorChat key={selected} modelId={selected} /></div>
        </div>
      )}
    </div>
  );
}

interface TurnMeta {
  model?: string;
  elapsed_s?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  cached_tokens?: number;
  note?: string;
}
interface Turn { role: string; content: string; reasoning?: string; meta?: TurnMeta }

function AdvisorChat({ modelId }: { modelId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  /** True while the model is reasoning but hasn't emitted answer text yet — the
   *  26B thinks for a beat before it writes, so we show a live 'thinking' pill
   *  instead of a dead spinner. */
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => { bottom.current?.scrollIntoView({ block: 'end' }); }, [turns, thinking]);

  const send = async () => {
    const message = input.trim();
    if (!message || busy) return;
    setInput(''); setError(null); setBusy(true); setThinking(true);
    const history = turns.map((t) => ({ role: t.role, content: t.content }));
    // Push the user turn plus an empty assistant turn the stream fills in place.
    setTurns((t) => [...t, { role: 'user', content: message }, { role: 'assistant', content: '' }]);
    const appendField = (field: 'content' | 'reasoning', delta: string) =>
      setTurns((t) => {
        const copy = t.slice();
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, [field]: (last[field] ?? '') + delta };
        return copy;
      });
    try {
      for await (const ev of api.streamAdvisorChat(modelId, message, history)) {
        if (ev.event === 'reasoning') {
          setThinking(true);                       // motion, but never shown as the answer
          if (ev.text) appendField('reasoning', ev.text);
        } else if (ev.event === 'text') {
          setThinking(false);
          if (ev.text) appendField('content', ev.text);
        } else if (ev.event === 'complete') {
          setThinking(false);
          setTurns((t) => {
            const copy = t.slice();
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = {
              ...last,
              content: last.content || (ev.note ?? ''),
              meta: {
                model: ev.model, elapsed_s: ev.elapsed_s,
                prompt_tokens: ev.usage?.prompt_tokens,
                completion_tokens: ev.usage?.completion_tokens,
                cached_tokens: ev.usage?.prompt_tokens_details?.cached_tokens,
                note: ev.note,
              },
            };
            return copy;
          });
        } else if (ev.event === 'error') {
          setThinking(false);
          setError(`${ev.kind ?? 'error'}: ${ev.detail ?? 'stream failed'}`);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false); setThinking(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 360 }}>
      <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Ask about this model</div>
      <p className="stat-sub" style={{ marginTop: 0 }}>
        Anchored to the facts on the left. It answers from those, and says unknown for anything they
        do not carry. Good openers: "is this a fit for agents?" or "are there better matches I own?"
      </p>
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '0.5rem 0' }}>
        {turns.map((t, i) => (
          <div key={i} style={{
            alignSelf: t.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '90%', padding: '0.5rem 0.75rem', borderRadius: 10, fontSize: '0.875rem',
            background: t.role === 'user' ? 'var(--accent-soft)' : 'var(--bg-subtle)',
          }}>
            {t.role === 'user'
              ? <span style={{ whiteSpace: 'pre-wrap' }}>{t.content}</span>
              : <>
                  {t.reasoning && <Thinking text={t.reasoning} live={!t.content} />}
                  {t.content ? <Markdown>{t.content}</Markdown> : null}
                  {t.meta && <TurnLine meta={t.meta} />}
                </>}
          </div>
        ))}
        {thinking && (
          <div style={{ alignSelf: 'flex-start', display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            <span className="stat-sub" style={{ fontStyle: 'italic' }}>thinking</span>
            <span className="stat-sub thinking-dots">…</span>
          </div>
        )}
        <div ref={bottom} />
      </div>
      {error && <Callout tone="bad">{error}</Callout>}
      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
        <input value={input} placeholder="Ask about this model…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void send(); }}
          style={{ flex: 1, fontFamily: 'inherit', fontSize: '0.875rem', padding: '0.5rem',
            border: '1px solid var(--border-strong)', borderRadius: 8 }} />
        <button type="button" className="btn-primary" disabled={busy || !input.trim()}
          onClick={() => void send()}>Send</button>
      </div>
    </div>
  );
}

/** The model's reasoning channel, collapsible. It is the model thinking out
 *  loud, NOT the answer, so it is visibly separate and labeled. While the model
 *  is still reasoning (no answer text yet) it auto-expands so the operator sees
 *  it working; once the answer arrives it collapses — unless the operator has
 *  clicked, in which case their choice sticks. */
function Thinking({ text, live }: { text: string; live: boolean }) {
  const [userSet, setUserSet] = useState<boolean | null>(null);
  const open = userSet ?? live;
  return (
    <div style={{ margin: '0 0 0.5rem' }}>
      <button type="button" onClick={() => setUserSet(!open)}
        style={{
          background: 'none', border: 'none', cursor: 'pointer', padding: 0,
          color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600,
          display: 'flex', alignItems: 'center', gap: '0.3rem', fontStyle: 'italic',
        }}>
        <span>{open ? '▾' : '▸'}</span>
        <span className={live ? 'thinking-dots' : undefined}>
          {live ? 'thinking…' : 'Thinking'}
        </span>
      </button>
      {open && (
        <div style={{
          marginTop: '0.35rem', paddingLeft: '0.6rem',
          borderLeft: '2px solid var(--border-color)',
          color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: 1.5,
          whiteSpace: 'pre-wrap', fontStyle: 'italic',
        }}>
          {text}
        </div>
      )}
    </div>
  );
}

/** The per-turn line (from the KAIROS observability note): tokens, wall-clock,
 *  model — so a slow turn is legible instead of a mystery. Cached-prompt hits
 *  are shown when the server reports them (prefix cache working). */
function TurnLine({ meta }: { meta: TurnMeta }) {
  const p = meta.prompt_tokens, c = meta.completion_tokens;
  const cached = meta.cached_tokens;
  const parts: string[] = [];
  if (p != null && c != null) parts.push(`${p.toLocaleString()}p + ${c.toLocaleString()}c`);
  if (cached) parts.push(`${cached.toLocaleString()} cached`);
  if (meta.elapsed_s != null) parts.push(`${meta.elapsed_s}s`);
  if (meta.model) parts.push(meta.model);
  if (!parts.length) return null;
  return (
    <p className="stat-sub" style={{ margin: '0.375rem 0 0', fontSize: '0.7rem', opacity: 0.75 }}>
      {parts.join('  ·  ')}
    </p>
  );
}

function AdvisorSettingsPanel() {
  const [settings, setSettings] = useState<AdvisorSettings | null>(null);
  const [persona, setPersona] = useState('');
  const [summaryPrompt, setSummaryPrompt] = useState('');
  const [primer, setPrimer] = useState('');
  const [provider, setProvider] = useState('gemini');
  const [searchModel, setSearchModel] = useState('');
  const [saved, setSaved] = useState<string | null>(null);
  const [searchTest, setSearchTest] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.advisorSettings().then((s) => {
      setSettings(s);
      setPersona(s.settings.persona_prompt);
      setSummaryPrompt(s.settings.summary_prompt);
      setPrimer(s.settings.knowledge_primer);
      setProvider(s.settings.web_search_provider);
      setSearchModel(s.settings.web_search_model);
    }).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  if (!settings) return null;

  const flash = (msg: string) => { setSaved(msg); setTimeout(() => setSaved(null), 1500); };
  const chooseModel = async (id: string) => {
    await api.updateAdvisorSettings({ analysis_model: id });
    refresh();
  };
  const saveInstructions = async () => {
    await api.updateAdvisorSettings({ persona_prompt: persona });
    flash('Instructions saved'); refresh();
  };
  const restoreInstructions = async () => {
    await api.updateAdvisorSettings({ persona_prompt: '__default__' });
    flash('Instructions restored'); refresh();
  };
  const chooseSummaryModel = async (id: string) => {
    await api.updateAdvisorSettings({ summary_model: id });
    refresh();
  };
  const saveSummaryInstructions = async () => {
    await api.updateAdvisorSettings({ summary_prompt: summaryPrompt });
    flash('Summary instructions saved'); refresh();
  };
  const restoreSummaryInstructions = async () => {
    await api.updateAdvisorSettings({ summary_prompt: '__default__' });
    flash('Summary instructions restored'); refresh();
  };
  const savePrimer = async () => {
    await api.updateAdvisorSettings({ knowledge_primer: primer });
    flash('Knowledge file saved'); refresh();
  };
  const restorePrimer = async () => {
    await api.updateAdvisorSettings({ knowledge_primer: '__default__' });
    flash('Knowledge file restored'); refresh();
  };
  const uploadPrimer = async (file: File) => {
    await api.uploadKnowledgeFile(file);
    flash('Knowledge file uploaded'); refresh();
  };
  const saveSearch = async () => {
    await api.updateAdvisorSettings({ web_search_provider: provider, web_search_model: searchModel });
    flash('Search saved'); refresh();
  };
  const runSearchTest = async () => {
    setSearchTest('Testing…');
    try {
      const r = await api.testWebSearch();
      setSearchTest(r.ok
        ? `Works. ${r.source_count} sources. Sample: ${r.sample?.slice(0, 120)}…`
        : `Failed: ${r.error}`);
    } catch (e) {
      setSearchTest(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
      <div>
        <label className="stat-sub">Analysis model (writes the Deep analysis)</label>
        <select value={settings.settings.analysis_model}
          onChange={(e) => void chooseModel(e.target.value)}
          style={{ display: 'block', marginTop: '0.25rem', fontFamily: 'inherit', fontSize: '0.9375rem',
            padding: '0.375rem 0.5rem', border: '1px solid var(--border-strong)', borderRadius: 8, minWidth: 320 }}>
          {settings.analysis_models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}{m.params_b != null ? ` (${m.params_b}B)` : ''}{m.not_recommended ? ' — not recommended, fabricates' : ''}
            </option>
          ))}
        </select>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          The big model behind the slow, deep teaching write-up. Defaults to the resident 26B: no
          extra cost, already vetted honest. Sub-2B models are kept but flagged, because below 2B
          this judgment-heavy role fabricates specs. (The quick Summary uses its own small model
          below.)
        </p>
      </div>

      {/* Summary (§1.10): its own model + instruction box, independent of the
          Analyze pair. A small non-reasoning model over a starved facts-card
          context — the fast path. */}
      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.875rem' }}>
        <label className="stat-sub">Summary model (the quick read)</label>
        <select value={settings.settings.summary_model}
          onChange={(e) => void chooseSummaryModel(e.target.value)}
          style={{ display: 'block', marginTop: '0.25rem', fontFamily: 'inherit', fontSize: '0.9375rem',
            padding: '0.375rem 0.5rem', border: '1px solid var(--border-strong)', borderRadius: 8, minWidth: 320 }}>
          {settings.summary_models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}{m.params_b != null ? ` (${m.params_b}B)` : ''}{m.warning ? ` — ${m.warning}` : ''}
            </option>
          ))}
        </select>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          Defaults to <code className="mono">Aevum-0.6B</code>. A small, fast, non-reasoning model
          is exactly right here: Summary only restates the facts card in plain words, and the starved
          context keeps it from wandering. This is the one place a sub-2B is recommended.
        </p>

        <label className="stat-sub" style={{ display: 'block', marginTop: '0.75rem' }}>
          Summary instructions
          {settings.settings.summary_prompt_is_default
            ? <span> (architect default)</span>
            : <span style={{ color: 'var(--status-good)' }}> (your custom version)</span>}
        </label>
        <textarea value={summaryPrompt} onChange={(e) => setSummaryPrompt(e.target.value)} rows={7}
          style={{ display: 'block', width: '100%', marginTop: '0.25rem', fontFamily: 'inherit',
            fontSize: '0.8125rem', padding: '0.5rem', border: '1px solid var(--border-strong)',
            borderRadius: 8, resize: 'vertical' }} />
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.375rem' }}>
          <button type="button" className="btn-secondary" onClick={() => void saveSummaryInstructions()}>Save summary instructions</button>
          {!settings.settings.summary_prompt_is_default && (
            <button type="button" className="btn-secondary" onClick={() => void restoreSummaryInstructions()}>Restore architect default</button>
          )}
          {saved?.startsWith('Summary instructions') && <span className="stat-sub" style={{ color: 'var(--status-good)' }}>{saved.replace('Summary instructions ', '')}</span>}
        </div>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          Governs only the quick Summary — separate from the Advisor instructions below, which drive
          the deep analysis and chat. Underneath, a locked floor always holds (facts card only, never
          looked anything up, no fabrication) and cannot be edited here.
        </p>
      </div>

      <div>
        <label className="stat-sub">
          Advisor instructions
          {settings.settings.persona_is_default
            ? <span> (architect default)</span>
            : <span style={{ color: 'var(--status-good)' }}> (your custom version)</span>}
        </label>
        <textarea value={persona} onChange={(e) => setPersona(e.target.value)} rows={8}
          style={{ display: 'block', width: '100%', marginTop: '0.25rem', fontFamily: 'inherit',
            fontSize: '0.8125rem', padding: '0.5rem', border: '1px solid var(--border-strong)',
            borderRadius: 8, resize: 'vertical' }} />
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.375rem' }}>
          <button type="button" className="btn-secondary" onClick={() => void saveInstructions()}>Save instructions</button>
          {!settings.settings.persona_is_default && (
            <button type="button" className="btn-secondary" onClick={() => void restoreInstructions()}>Restore architect default</button>
          )}
          {saved === 'Instructions saved' && <span className="stat-sub" style={{ color: 'var(--status-good)' }}>Saved</span>}
          {saved === 'Instructions restored' && <span className="stat-sub" style={{ color: 'var(--status-good)' }}>Restored</span>}
        </div>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          The advisor's full behaviour: how it teaches, its three sources of truth, when to search,
          how it recommends. Edit freely and it persists. Underneath, a locked safety rule (never
          state a model-specific fact from memory, never fake a citation) always holds and cannot be
          edited here.
        </p>
      </div>

      <div>
        <label className="stat-sub">
          Knowledge file — the primer on how local models work
          {settings.settings.knowledge_primer_active
            ? (settings.settings.knowledge_primer_is_default
                ? <span style={{ color: 'var(--status-good)' }}> (architect default attached)</span>
                : <span style={{ color: 'var(--status-good)' }}> (your file attached)</span>)
            : <span> (none)</span>}
        </label>
        <textarea value={primer} onChange={(e) => setPrimer(e.target.value)} rows={6}
          placeholder="The primer on quantization, MoE, context, hardware fit… injected as the authoritative source for concept explanations."
          style={{ display: 'block', width: '100%', marginTop: '0.25rem', fontFamily: 'inherit',
            fontSize: '0.75rem', padding: '0.5rem', border: '1px solid var(--border-strong)',
            borderRadius: 8, resize: 'vertical' }} />
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.375rem', flexWrap: 'wrap' }}>
          <button type="button" className="btn-secondary" onClick={() => void savePrimer()}>Save edits</button>
          <label className="btn-secondary" style={{ cursor: 'pointer', margin: 0 }}>
            Upload a file…
            <input type="file" accept=".md,.txt,text/markdown,text/plain" style={{ display: 'none' }}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void uploadPrimer(f); e.target.value = ''; }} />
          </label>
          {!settings.settings.knowledge_primer_is_default && (
            <button type="button" className="btn-secondary" onClick={() => void restorePrimer()}>Restore architect default</button>
          )}
          {saved?.startsWith('Knowledge file') && <span className="stat-sub" style={{ color: 'var(--status-good)' }}>{saved.replace('Knowledge file ', '')}</span>}
        </div>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          The advisor teaches concepts (quant, MoE, fit) from THIS document, not its own recall.
          Paste, edit, or upload a new file to update it as the architect evolves it. Model-specific
          facts still come only from the computed facts, HuggingFace, and web search.
        </p>
      </div>

      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.875rem' }}>
        <label className="stat-sub">Web search (grounded lookup)</label>
        <div style={{ fontSize: '0.875rem', marginTop: '0.25rem' }}>
          {settings.web_search.enabled ? (
            <span style={{ color: 'var(--status-good)' }}>● On (key configured)</span>
          ) : (
            <span className="stat-sub">
              ○ Off. Set <code className="mono">HELM_WEB_SEARCH_KEY</code> in the stack's .env
              (a Gemini grounding key, separate from any model key) to let the advisor research
              models live.
            </span>
          )}
        </div>

        {/* Editable provider + model. The KEY stays in .env; these are the
            non-secret settings the operator swaps as newer grounding models
            ship — no redeploy. */}
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label className="stat-sub" style={{ display: 'block' }}>Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}
              style={{ fontFamily: 'inherit', fontSize: '0.875rem', padding: '0.375rem 0.5rem',
                border: '1px solid var(--border-strong)', borderRadius: 8, marginTop: '0.25rem' }}>
              <option value="gemini">Gemini (Google grounding)</option>
              <option value="tavily">Tavily</option>
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 180 }}>
            <label className="stat-sub" style={{ display: 'block' }}>
              Search model
              {!settings.settings.web_search_model_is_default && <span> (custom)</span>}
            </label>
            <input value={searchModel} onChange={(e) => setSearchModel(e.target.value)}
              placeholder="gemini-2.5-flash"
              style={{ width: '100%', fontFamily: 'inherit', fontSize: '0.875rem',
                padding: '0.375rem 0.5rem', border: '1px solid var(--border-strong)',
                borderRadius: 8, marginTop: '0.25rem' }} />
          </div>
          <button type="button" className="btn-secondary" onClick={() => void saveSearch()}>Save</button>
          {settings.web_search.enabled && (
            <button type="button" className="btn-secondary" onClick={() => void runSearchTest()}>Test search</button>
          )}
        </div>
        <div style={{ marginTop: '0.375rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {saved === 'Search saved' && <span className="stat-sub" style={{ color: 'var(--status-good)' }}>Saved</span>}
          {searchTest && <span className="stat-sub">{searchTest}</span>}
        </div>
        <p className="stat-sub" style={{ marginTop: '0.375rem' }}>
          The model must support the provider's grounding. Swap it here when a newer grounding
          model ships — it takes effect immediately, no redeploy. The key stays in the .env.
        </p>
      </div>
    </div>
  );
}
