/**
 * Login / first-run setup.
 *
 * One component for both because they differ only in copy and in which
 * endpoint they hit — and because the operator should never be able to reach
 * a login form on a box that hasn't been set up (that's how you get someone
 * typing their intended password into a form that's silently a no-op).
 */
import { useState } from 'react';
import { ApiError, api } from '../api/client';

const MIN_PASSWORD_LENGTH = 12;

export default function Login({
  initialized, schemeSource, onDone,
}: {
  initialized: boolean;
  schemeSource: string;
  onDone: () => void;
}) {
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const isSetup = !initialized;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (isSetup && password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    if (isSetup && password.length < MIN_PASSWORD_LENGTH) {
      setError(`Use at least ${MIN_PASSWORD_LENGTH} characters.`);
      return;
    }
    setBusy(true);
    try {
      if (isSetup) await api.setup(password);
      else await api.login(password);
      onDone();
    } catch (err) {
      // Say which failure. A 409 ("already set up") and a 401 ("wrong
      // password") send the operator in completely different directions.
      if (err instanceof ApiError) {
        setError(
          typeof err.detail === 'string' ? err.detail : `Failed (HTTP ${err.status}).`,
        );
      } else {
        setError(String(err));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: '1rem' }}>
      <form onSubmit={submit} className="card" style={{ width: '100%', maxWidth: 400 }}>
        <div style={{ marginBottom: '1rem' }}>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700 }}>HELM</h1>
          <p className="stat-sub" style={{ marginTop: '0.25rem' }}>
            {isSetup ? 'Create an operator password to finish setup.' : 'AI fleet control plane'}
          </p>
        </div>

        {/* Behind NPM this should never be 'none'. If it is, the proxy isn't
            forwarding X-Forwarded-Proto and the session cookie won't be
            marked Secure — worth saying out loud rather than silently
            downgrading. Harmless in local dev over http. */}
        {schemeSource === 'none' && (
          <div
            style={{
              background: 'var(--status-warn-soft)', color: 'var(--status-warn)',
              borderRadius: 8, padding: '0.5rem 0.625rem',
              fontSize: '0.75rem', marginBottom: '0.75rem', lineHeight: 1.5,
            }}
          >
            Connection is not detected as HTTPS, so the session cookie won't be
            marked Secure. Expected on <span className="mono">localhost</span>.
            If you're on the real host, your proxy isn't forwarding{' '}
            <span className="mono">X-Forwarded-Proto</span>.
          </div>
        )}

        <div className="field" style={{ marginBottom: '0.75rem' }}>
          <label htmlFor="pw">Password</label>
          <input
            id="pw"
            type="password"
            autoFocus
            autoComplete={isSetup ? 'new-password' : 'current-password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        {isSetup && (
          <div className="field" style={{ marginBottom: '0.75rem' }}>
            <label htmlFor="pw2">Confirm password</label>
            <input
              id="pw2"
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          </div>
        )}

        {isSetup && (
          // There is no recovery. Say it BEFORE they commit, not in a docs
          // page they'll read after they're locked out.
          <p className="stat-sub" style={{ marginBottom: '0.75rem', lineHeight: 1.5 }}>
            At least {MIN_PASSWORD_LENGTH} characters. There is no password
            recovery — store it in a password manager. To reset, delete{' '}
            <span className="mono">auth.json</span> from HELM's data directory
            and restart.
          </p>
        )}

        {error && (
          <div style={{
            background: 'var(--status-bad-soft)', color: 'var(--status-bad)',
            borderRadius: 8, padding: '0.5rem 0.625rem',
            fontSize: '0.8125rem', marginBottom: '0.75rem',
          }}>
            {error}
          </div>
        )}

        <button type="submit" className="btn-primary" disabled={busy || !password} style={{ width: '100%' }}>
          {busy ? 'Working…' : isSetup ? 'Create password' : 'Log in'}
        </button>
      </form>
    </div>
  );
}
