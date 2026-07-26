/**
 * The load dialog. Spec §5.4, §5.3.
 *
 * Two rules shape this whole component:
 *
 * 1. **The operator thinks in "8 slots of 64K". Lemonade thinks in
 *    `ctx_size: 524288`.** HELM does the multiplication; the total is shown as
 *    a computed consequence, never as an input. The maths happens on the
 *    server (`domain.load_plan`) and arrives as `computed_ctx_size` — this
 *    file never multiplies anything, because a second implementation of Rule 1
 *    is how the two drift apart.
 *
 * 2. **Slot controls are HIDDEN for non-llamacpp recipes**, not disabled.
 *    §5.3: "recipe flm -> hide slot controls entirely. Showing a slot field
 *    would be a lie." The NPU is serial; a greyed-out slot box still implies
 *    the concept exists.
 *
 * Every keystroke re-plans against the live fleet, so the NPU-eviction warning
 * appears while the operator is still deciding.
 */
import { useCallback, useEffect, useState } from 'react';
import { ApiError, api, type LibraryEntry, type LoadRequest, type Plan } from '../api/client';
import { Callout, Issues, Modal } from './common';

const SLOTTED_RECIPES = new Set(['llamacpp']);

/** Prefill values, read off the model's current or saved configuration. */
export interface LoadPrefill {
  slots?: number | null;
  ctxPerSlot?: number | null;
  backend?: string | null;
  temperature?: number | null;
  topK?: number | null;
  topP?: number | null;
  minP?: number | null;
  repeatPenalty?: number | null;
  extraArgs?: string;
}

/** The sampling fields, described once. Rendered as a row of number inputs and
 *  composed back into llamacpp_args server-side (F4: HELM owns the spelling). */
const SAMPLING_FIELDS = [
  { key: 'temperature', label: 'Temperature', step: '0.05', placeholder: 'e.g. 0.7' },
  { key: 'topK', label: 'Top-k', step: '1', placeholder: 'e.g. 40' },
  { key: 'topP', label: 'Top-p', step: '0.01', placeholder: 'e.g. 0.9' },
  { key: 'minP', label: 'Min-p', step: '0.01', placeholder: 'e.g. 0.05' },
  { key: 'repeatPenalty', label: 'Repeat penalty', step: '0.01', placeholder: 'e.g. 1.1' },
] as const;

export default function LoadDialog({ entry, prefill, mode = 'load', onClose, onDone }: {
  entry: LibraryEntry;
  prefill?: LoadPrefill;
  /** 'reconfigure' means the model is already resident, so applying reloads it. */
  mode?: 'load' | 'reconfigure';
  onClose: () => void;
  onDone: () => void;
}) {
  const showSlots = SLOTTED_RECIPES.has(entry.recipe ?? '');

  const [slots, setSlots] = useState<string>(
    prefill?.slots != null ? String(prefill.slots) : '');
  const [ctxPerSlot, setCtxPerSlot] = useState<string>(
    prefill?.ctxPerSlot != null ? String(prefill.ctxPerSlot) : '');
  const [backend, setBackend] = useState<string>(prefill?.backend ?? '');
  const [pinned, setPinned] = useState(entry.pinned);
  const [extraArgs, setExtraArgs] = useState(prefill?.extraArgs ?? '');
  const [saveOptions, setSaveOptions] = useState(true);

  // Sampling fields as strings (empty = "leave unset"), keyed by SAMPLING_FIELDS.
  const [sampling, setSampling] = useState<Record<string, string>>(() => ({
    temperature: prefill?.temperature != null ? String(prefill.temperature) : '',
    topK: prefill?.topK != null ? String(prefill.topK) : '',
    topP: prefill?.topP != null ? String(prefill.topP) : '',
    minP: prefill?.minP != null ? String(prefill.minP) : '',
    repeatPenalty: prefill?.repeatPenalty != null ? String(prefill.repeatPenalty) : '',
  }));

  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  // An empty or non-numeric sampling field means "leave it unset" — null, not
  // zero. A zero min-p is a real, different instruction from "don't set min-p".
  const samplingValue = (key: string): number | null => {
    if (!showSlots) return null;
    const raw = sampling[key]?.trim();
    if (!raw) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };

  const request = useCallback((): LoadRequest => ({
    model_name: entry.model_id,
    recipe: entry.recipe,
    slots: showSlots && slots ? Number(slots) : null,
    ctx_per_slot: ctxPerSlot ? Number(ctxPerSlot) : null,
    llamacpp_backend: showSlots && backend ? backend : null,
    pinned,
    temperature: samplingValue('temperature'),
    top_k: samplingValue('topK'),
    top_p: samplingValue('topP'),
    min_p: samplingValue('minP'),
    repeat_penalty: samplingValue('repeatPenalty'),
    extra_args: extraArgs,
    save_options: saveOptions,
  }), [entry, showSlots, slots, ctxPerSlot, backend, pinned, sampling, extraArgs, saveOptions]);

  // Re-plan as the form changes. Debounced: the planner reads /health on every
  // call, and the point is to avoid making the dashboard a load source on the
  // box it watches — that applies to the dialog too.
  useEffect(() => {
    const id = setTimeout(() => {
      api.planLoad(request()).then(setPlan).catch(() => setPlan(null));
    }, 250);
    return () => clearTimeout(id);
  }, [request]);

  const submit = async (confirm: boolean) => {
    setBusy(true);
    setFailure(null);
    try {
      const result = await api.load(request(), confirm);
      if (!result.performed) {
        // The upstream refused. Keep the KIND — a rejected key, an
        // exclusivity refusal and a timeout need different reactions.
        setFailure(`${result.error_kind}: ${result.error_detail}`);
        return;
      }
      if (result.verified === false) {
        // Lemonade accepted the reload but came back running DIFFERENT options
        // than we sent (the --parallel-drop case). Do NOT report success or
        // close: a silent no-op is exactly what this dialog must stop doing.
        setFailure(result.verify_note
          ?? 'The reload was accepted, but the running options do not match what you set — the change did not take.');
        return;
      }
      onDone();
    } catch (e) {
      if (e instanceof ApiError && (e.status === 409 || e.status === 422)) {
        // Expected flows, not bugs: 409 = needs confirmation, 422 = blocked.
        setPlan(e.detail as Plan);
        return;
      }
      setFailure(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const blocked = plan?.blocked ?? false;
  const needsConfirm = plan?.needs_confirmation ?? false;

  const verb = mode === 'reconfigure' ? 'Apply' : 'Load';

  return (
    <Modal
      title={`${mode === 'reconfigure' ? 'Settings' : 'Load'} — ${entry.model_id}`}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button
            type="button"
            className={needsConfirm ? 'btn-danger' : 'btn-primary'}
            disabled={busy || blocked}
            onClick={() => submit(needsConfirm)}
            title={blocked ? 'Fix the errors above first' : undefined}
          >
            {busy ? 'Working…' : needsConfirm ? `${verb} anyway` : verb}
          </button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
        {failure && <Callout tone="bad" title="Lemonade refused">{failure}</Callout>}

        {mode === 'reconfigure' && (
          // Lemonade has no "change options without restarting" call, so
          // applying goes through `load`, which respawns the backend process.
          // Say that plainly rather than let it be a surprise.
          <Callout tone="warn" title="Applying reloads the model">
            There is no way to change a running model's options in place — Lemonade applies them
            by reloading it. It drops for roughly 10-20 seconds and any request in flight against
            it fails.
          </Callout>
        )}

        {showSlots ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <label className="field">
              <span>Slots (concurrent requests)</span>
              <input type="number" min={1} value={slots} placeholder="auto"
                onChange={(e) => setSlots(e.target.value)} />
            </label>
            <label className="field">
              <span>Context per slot</span>
              <input type="number" min={1} value={ctxPerSlot} placeholder="auto"
                onChange={(e) => setCtxPerSlot(e.target.value)} />
            </label>
          </div>
        ) : (
          // Hidden, not disabled. Rule 2 / §5.3.
          <Callout tone="info">
            <strong>{entry.recipe}</strong> has no slot concept
            {entry.recipe === 'flm' && ' — the NPU processes one request at a time regardless of context'}.
            Slot controls are hidden rather than greyed out, because showing them at all would
            imply the setting exists.
          </Callout>
        )}

        {/* The total, as a consequence. Computed server-side by the same
            function the dashboard's Concurrency column uses. */}
        <div className="card" style={{ background: 'var(--bg-subtle)', padding: '0.625rem 0.75rem' }}>
          <div className="stat-sub">Total ctx_size HELM will send (read-only)</div>
          <div className="mono tabular" style={{ fontSize: '1.05rem', fontWeight: 600 }}>
            {plan?.computed_ctx_size != null
              ? plan.computed_ctx_size.toLocaleString()
              : <span className="unknown">auto</span>}
          </div>
          <div className="stat-sub" style={{ marginTop: '0.25rem' }}>
            Lemonade's ctx_size is the TOTAL across slots. You set the per-slot value; HELM
            multiplies. You should never have to do this sum.
          </div>
        </div>

        {showSlots && (
          <label className="field">
            <span>Backend</span>
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="">auto (currently rocm on this box)</option>
              <option value="rocm">rocm</option>
              <option value="vulkan">vulkan</option>
              <option value="cpu">cpu</option>
            </select>
          </label>
        )}

        {showSlots && (
          <div className="field">
            <span>Sampling (leave blank to use the model's own default)</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.5rem' }}>
              {SAMPLING_FIELDS.map((f) => (
                <label key={f.key} className="field" style={{ gap: '0.25rem' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>{f.label}</span>
                  <input
                    type="number"
                    step={f.step}
                    value={sampling[f.key]}
                    placeholder={f.placeholder}
                    onChange={(e) =>
                      setSampling((s) => ({ ...s, [f.key]: e.target.value }))}
                  />
                </label>
              ))}
            </div>
            <span className="stat-sub">
              HELM writes these into <code>llamacpp_args</code> for you — don't also type them
              below. Setting a value in both places is rejected, because the two would silently
              disagree.
            </span>
          </div>
        )}

        <label className="field">
          <span>Extra llama.cpp arguments (advanced)</span>
          <input value={extraArgs} placeholder="--flash-attn on --no-mmap"
            onChange={(e) => setExtraArgs(e.target.value)} />
          <span className="stat-sub">
            Context and device arguments are reserved — Lemonade wires those itself and will
            reject them. HELM checks before sending.
          </span>
        </label>

        <div style={{ display: 'flex', gap: '1.25rem' }}>
          <label style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', fontSize: '0.875rem' }}>
            <input type="checkbox" checked={pinned} onChange={(e) => setPinned(e.target.checked)} />
            Pinned (exempt from eviction)
          </label>
          <label style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', fontSize: '0.875rem' }}>
            <input type="checkbox" checked={saveOptions}
              onChange={(e) => setSaveOptions(e.target.checked)} />
            Save as default
          </label>
        </div>
        <div className="stat-sub" style={{ marginTop: '-0.5rem' }}>
          Without "save as default" the model runs with these options but persists nothing — it
          reverts on the next lemond restart, and the dashboard will show the drift.
        </div>

        {plan && <Issues issues={plan.issues} />}
      </div>
    </Modal>
  );
}
