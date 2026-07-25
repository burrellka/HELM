/**
 * App shell + auth gate.
 *
 * The shell owns auth and the upstream banner; the Dashboard owns Phase 1's
 * polling and rendering. No router yet — there is one page. Phase 2's control
 * views and Phase 3's library are what earn a route table, and adding one now
 * would be scaffolding for a shape nobody has settled.
 */
import { useCallback, useEffect, useState } from 'react';
import { ApiError, api, type AuthStatus, type SystemConfig } from './api/client';
import Advisor from './pages/Advisor';
import DownloadIndicator from './components/DownloadIndicator';
import Dashboard from './pages/Dashboard';
import Library from './pages/Library';
import Logs from './pages/Logs';
import Login from './pages/Login';
import ModelDetailPage from './pages/ModelDetail';
import Operations from './pages/Operations';

type Tab = 'dashboard' | 'library' | 'advisor' | 'logs' | 'operations';

const TABS: { id: Tab; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'library', label: 'Library' },
  { id: 'advisor', label: 'Advisor' },
  { id: 'logs', label: 'Logs' },
  { id: 'operations', label: 'Operations' },
];

export default function App() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>('dashboard');
  /** Set when the dashboard sends the operator to a specific model's settings.
   *  The editor lives in the Library because it needs the catalog record — the
   *  trained context ceiling that /health does not carry. */
  const [focusModelId, setFocusModelId] = useState<string | null>(null);
  /** Set when the operator opens a model's dedicated detail page. Non-null takes
   *  over the main area (over whatever tab is active) until they go Back. */
  const [detailModelId, setDetailModelId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const status = await api.authStatus();
      setAuth(status);
      if (status.authenticated || status.auth_disabled) {
        try {
          setConfig(await api.systemConfig());
        } catch (e) {
          // A 401 here means the cookie died between the two calls. Fall
          // back to the login gate rather than rendering a broken shell.
          if (e instanceof ApiError && e.status === 401) {
            setAuth({ ...status, authenticated: false });
          } else {
            throw e;
          }
        }
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  if (loading && !auth) {
    return <Centered>Loading…</Centered>;
  }
  if (auth && !auth.authenticated && !auth.auth_disabled) {
    return <Login initialized={auth.initialized} schemeSource={auth.scheme_source} onDone={refresh} />;
  }

  return (
    <div style={{ minHeight: '100vh' }}>
      <Header config={config} onLogout={async () => { await api.logout(); await refresh(); }} />
      <Nav tab={tab} onSelect={(t) => { setFocusModelId(null); setDetailModelId(null); setTab(t); }} />
      <main style={{ padding: '1.5rem', maxWidth: 1400, margin: '0 auto' }}>
        {config && <UpstreamBanner config={config} />}
        <div style={{ marginTop: '1rem' }}>
          {/* `management_enabled` gates every write control. When it's false the
              buttons are ABSENT, not disabled — an Unload that 401s is worse
              than no Unload at all (URSA 1.1.8). */}
          {detailModelId ? (
            <ModelDetailPage
              modelId={detailModelId}
              managementEnabled={config?.upstream.management_enabled ?? false}
              onBack={() => setDetailModelId(null)}
              onConfigure={(modelName) => { setDetailModelId(null); setFocusModelId(modelName); setTab('library'); }}
              onUpdate={(modelName) => { setDetailModelId(null); setFocusModelId(modelName); setTab('library'); }}
            />
          ) : (
          <>
          {tab === 'dashboard' && (
            <Dashboard
              managementEnabled={config?.upstream.management_enabled ?? false}
              onConfigure={(modelName) => { setFocusModelId(modelName); setTab('library'); }}
              onUpdate={(modelName) => { setFocusModelId(modelName); setTab('library'); }}
              onOpenDetail={(modelName) => setDetailModelId(modelName)}
            />
          )}
          {tab === 'library' && (
            <Library
              key={focusModelId ?? 'all'}
              managementEnabled={config?.upstream.management_enabled ?? false}
              focusModelId={focusModelId}
            />
          )}
          {tab === 'advisor' && <Advisor />}
          {tab === 'logs' && <Logs />}
          {tab === 'operations' && (
            <Operations managementEnabled={config?.upstream.management_enabled ?? false} />
          )}
          </>
          )}
        </div>
      </main>
    </div>
  );
}

function Header({ config, onLogout }: { config: SystemConfig | null; onLogout: () => void }) {
  return (
    <header
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0.75rem 1.5rem',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border-color)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.625rem' }}>
          <span style={{ fontWeight: 700, fontSize: '1.0625rem' }}>HELM</span>
          <span className="stat-sub">Homelab Engine for Lemonade Management</span>
          {config && <span className="stat-sub mono">v{config.version}</span>}
        </div>
        <DownloadIndicator />
      </div>
      <button type="button" className="btn-secondary" onClick={onLogout}>Log out</button>
    </header>
  );
}

function Nav({ tab, onSelect }: { tab: Tab; onSelect: (t: Tab) => void }) {
  return (
    <nav style={{
      display: 'flex', gap: '0.25rem', padding: '0 1.5rem',
      background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-color)',
    }}>
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => onSelect(t.id)}
          // Clicking Library directly means "show me everything", so drop any
          // focus left over from a dashboard Settings click.

          style={{
            fontFamily: 'inherit', fontSize: '0.875rem', fontWeight: 500, cursor: 'pointer',
            padding: '0.625rem 0.875rem', border: 'none', background: 'none',
            color: tab === t.id ? 'var(--accent-text)' : 'var(--text-secondary)',
            borderBottom: `2px solid ${tab === t.id ? 'var(--accent-primary)' : 'transparent'}`,
            marginBottom: -1,
          }}
        >{t.label}</button>
      ))}
    </nav>
  );
}

/**
 * The upstream banner.
 *
 * Deliberately says WHICH failure it is, never just "error". A dead lemond,
 * a rejected key and a missing key need three different operator actions.
 * (See UpstreamErrorKind in the backend.)
 */
function UpstreamBanner({ config }: { config: SystemConfig }) {
  const u = config.upstream;

  if (!u.reachable) {
    return (
      <Banner tone="bad">
        <strong>Lemonade unreachable</strong> — {u.lemonade_url}
        <div className="stat-sub" style={{ marginTop: '0.25rem' }}>
          {u.error_kind}: {u.error_detail}
        </div>
      </Banner>
    );
  }
  if (!u.management_enabled) {
    // Read-only is a legitimate mode (a perfectly good Phase 1). Say so
    // plainly and hide the write controls — do NOT render buttons that 401.
    return (
      <Banner tone="warn">
        <strong>Read-only</strong> — no Lemonade admin key configured, so
        load/unload/pin/delete are unavailable. Set
        {' '}<code className="mono">HELM_LEMONADE_ADMIN_API_KEY</code> in the
        stack's .env to enable them.
      </Banner>
    );
  }
  return (
    <Banner tone="good">
      Connected to Lemonade at <span className="mono">{u.lemonade_url}</span> — management enabled.
    </Banner>
  );
}

function Banner({ tone, children }: { tone: 'good' | 'warn' | 'bad'; children: React.ReactNode }) {
  const bg = {
    good: 'var(--status-good-soft)', warn: 'var(--status-warn-soft)', bad: 'var(--status-bad-soft)',
  }[tone];
  const fg = {
    good: 'var(--status-good)', warn: 'var(--status-warn)', bad: 'var(--status-bad)',
  }[tone];
  return (
    <div style={{
      background: bg, color: fg, border: `1px solid ${fg}33`,
      borderRadius: 8, padding: '0.625rem 0.875rem', fontSize: '0.875rem',
    }}>
      {children}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      minHeight: '100vh', display: 'grid', placeItems: 'center',
      color: 'var(--text-secondary)',
    }}>
      {children}
    </div>
  );
}
