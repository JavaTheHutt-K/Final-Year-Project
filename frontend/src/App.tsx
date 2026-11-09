import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Link,
  NavLink,
  Outlet,
  useLocation,
} from "react-router-dom";
import {
  getApiErrorMessage,
  getConfig,
  getResults,
  getRunStatus,
  getTrainingHistory,
  hasAdminSession,
  login,
  logout,
} from "./api";
import {
  DashboardContext,
  type DashboardContextValue,
} from "./DashboardContext";
import type {
  ExperimentConfig,
  ExperimentResults,
  RunStatus,
  TrainingHistoryEntry,
} from "./types";

const EMPTY_STATUS: RunStatus = { status: "idle" };
const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "⌂", end: true },
  { to: "/live-traffic", label: "Live Traffic", icon: "▥" },
  { to: "/detections", label: "Detections", icon: "⚠" },
  { to: "/insights", label: "Insights", icon: "✦" },
  { to: "/model-performance", label: "Model Performance", icon: "↗" },
  { to: "/devices", label: "Devices", icon: "▣" },
  { to: "/clients", label: "Clients", icon: "♧" },
  { to: "/reports", label: "Reports", icon: "▤" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];
const PAGE_TITLES: Record<string, string> = {
  "/": "Dashboard",
  "/dataset": "Live Traffic",
  "/live-traffic": "Live Traffic",
  "/training": "Clients",
  "/clients": "Clients",
  "/results": "Detections",
  "/detections": "Detections",
  "/insights": "Insights",
  "/reports": "Reports",
  "/adaptation": "Model Performance",
  "/model-performance": "Model Performance",
  "/devices": "Devices",
  "/settings": "Settings",
};

function AppLayout({ onLogout }: { onLogout: () => void }) {
  const [config, setConfig] = useState<ExperimentConfig | null>(null);
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);
  const [status, setStatus] = useState<RunStatus>(EMPTY_STATUS);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTestSet, setSelectedTestSet] = useState("");
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    localStorage.getItem("dashboard-theme") === "dark" ? "dark" : "light",
  );
  const [autoRefresh, setAutoRefresh] = useState(
    () => localStorage.getItem("dashboard-auto-refresh") !== "false",
  );
  const [refreshInterval, setRefreshInterval] = useState(() => {
    const savedInterval = Number(localStorage.getItem("dashboard-refresh-interval"));
    return [15, 30, 60].includes(savedInterval) ? savedInterval : 30;
  });
  const location = useLocation();

  const refreshDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextConfig, nextResults, nextHistory] = await Promise.all([
        getConfig(),
        getResults(),
        getTrainingHistory(),
      ]);
      setConfig(nextConfig);
      setResults(nextResults);
      setHistory(
        nextHistory.length
          ? nextHistory
          : nextResults.training_history ?? [],
      );
      const availableTestSets = Object.keys(nextResults.metrics?.federated ?? {});
      setSelectedTestSet((current) =>
        current === "" || availableTestSets.includes(current) ? current : "",
      );
    } catch (requestError) {
      setError(getApiErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const nextStatus = await getRunStatus();
      setStatus(nextStatus);
      setRunning(nextStatus.status === "queued" || nextStatus.status === "running");
    } catch (requestError) {
      setStatus({
        status: "unavailable",
        error: `Could not read run status: ${getApiErrorMessage(requestError)}`,
      });
    }
  }, []);

  useEffect(() => {
    void refreshDashboard();
    void refreshStatus();
  }, [refreshDashboard, refreshStatus]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("dashboard-theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("dashboard-auto-refresh", String(autoRefresh));
    localStorage.setItem("dashboard-refresh-interval", String(refreshInterval));
  }, [autoRefresh, refreshInterval]);

  useEffect(() => {
    if (!running && !autoRefresh) return undefined;
    const timer = window.setInterval(() => {
      void refreshStatus();
      if (autoRefresh) void refreshDashboard();
    }, refreshInterval * 1000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, refreshInterval, running, refreshDashboard, refreshStatus]);

  const testSets = useMemo(
    () => {
      const evaluatedTestSets = Object.keys(results?.metrics?.federated ?? {});
      if (evaluatedTestSets.length) return evaluatedTestSets;
      const configuredTestSet = config?.split?.test_set_name;
      return configuredTestSet ? [configuredTestSet] : [];
    },
    [results, config],
  );
  const value = useMemo<DashboardContextValue>(
    () => ({
      config,
      results,
      history,
      status,
      loading,
      running,
      error,
      testSets,
      selectedTestSet,
      setSelectedTestSet,
      theme,
      setTheme,
      autoRefresh,
      setAutoRefresh,
      refreshInterval,
      setRefreshInterval,
      refreshDashboard,
      refreshStatus,
    }),
    [
      config,
      results,
      history,
      status,
      loading,
      running,
      error,
      testSets,
      selectedTestSet,
      theme,
      autoRefresh,
      refreshInterval,
      refreshDashboard,
      refreshStatus,
    ],
  );
  const pageTitle = PAGE_TITLES[location.pathname] ?? "Dashboard";
  const totalClassificationErrors = Object.values(results?.metrics?.federated ?? {}).reduce(
    (total, metrics) => total + metrics.confusion_matrix.fp + metrics.confusion_matrix.fn,
    0,
  );

  return (
    <DashboardContext.Provider value={value}>
      <div className={`app-shell theme-${theme}`}>
        <aside className="sidebar">
          <Link className="brand" to="/" aria-label="Intrusion Detection dashboard">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24"><path d="M12 2.5 20 6v5.5c0 5.2-3.4 8.6-8 10.1-4.6-1.5-8-4.9-8-10.1V6l8-3.5Z" /><path d="m8.6 12 2.1 2.1 4.8-4.8" /></svg>
            </span>
            <span className="brand-copy">
              <strong>Intrusion Detection</strong>
              <small>INTRUSION DETECTION</small>
            </span>
          </Link>
          <div className="sidebar-label">DASHBOARD</div>
          <nav className="side-nav" aria-label="Main navigation">
            {NAV_ITEMS.map((item) => (
              <NavLink
                className={({ isActive }) =>
                  `nav-link ${isActive ? "active" : ""}`
                }
                end={item.end}
                key={item.to}
                to={item.to}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="sidebar-dataset">
              <span className="dataset-icon">◎</span>
              <div><strong>CICIDS2017</strong><small>Network traffic dataset</small></div>
            </div>
            <div className="sidebar-footnote">Federated and centralised model monitoring</div>
          </div>
        </aside>

        <main className="main-content">
          <header className="topbar">
            <div className="topbar-title">{pageTitle}</div>
            <div className="topbar-actions">
              {error && (
                <span className="connection-error" role="alert" title={error}>
                  <i />API unavailable
                </span>
              )}
              <label className="date-filter" aria-label="Select test set">
                <span className="calendar-icon">▣</span>
                <select
                  aria-label="Test set"
                  title="Filter dashboard data by test set"
                  value={selectedTestSet}
                  onChange={(event) => setSelectedTestSet(event.target.value)}
                >
                  <option value="">All test sets</option>
                  {testSets.map((testSet) => <option key={testSet} value={testSet}>{testSet}</option>)}
                </select>
              </label>
              <button
                className={`topbar-icon-button ${loading ? "is-refreshing" : ""}`}
                aria-label={loading ? "Refreshing dashboard data" : "Refresh dashboard data"}
                aria-busy={loading}
                title={loading ? "Refreshing dashboard data" : "Refresh dashboard data"}
                onClick={() => void refreshDashboard()}
                disabled={loading}
              >
                <span aria-hidden="true">↻</span>
              </button>
              <Link
                className="notification-button"
                to="/detections"
                aria-label={totalClassificationErrors > 0
                  ? `View detection results; ${totalClassificationErrors.toLocaleString()} classification errors`
                  : "View detection results"}
                title={totalClassificationErrors > 0
                  ? `${totalClassificationErrors.toLocaleString()} classification errors`
                  : "No classification errors recorded"}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></svg>
                {totalClassificationErrors > 0 && <i />}
              </Link>
              <Link className="profile-link" to="/settings" aria-label="Open dashboard settings" title="Dashboard settings">
                <span className="avatar">IDS</span><span className="profile-label">Settings</span>
              </Link>
              <button
                className="topbar-icon-button"
                type="button"
                onClick={onLogout}
                aria-label="Log out"
                title="Log out"
              >
                ↪
              </button>
            </div>
          </header>
          <div className="content-wrap">
            <Outlet />
          </div>
        </main>
      </div>
    </DashboardContext.Provider>
  );
}

function LoginScreen({
  notice,
  onLogin,
}: {
  notice: string;
  onLogin: (password: string) => Promise<void>;
}) {
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(notice);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => setError(notice), [notice]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await onLogin(password);
    } catch (loginError) {
      setError(getApiErrorMessage(loginError));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-screen">
      <div className="login-background" aria-hidden="true" />
      <section className="login-intro" aria-label="Adaptive IDS">
        <div className="login-brand">
          <span className="login-brand-mark">IDS</span>
          <span>Adaptive <strong>IDS</strong></span>
        </div>
        <p className="login-eyebrow">INTELLIGENT THREAT MONITORING</p>
        <h1>Stay ahead of what’s happening on your network.</h1>
        <p className="login-intro-copy">
          One secure workspace for intrusion insights, model performance, and
          adaptive detection.
        </p>
        <div className="login-feature-list" aria-label="Workspace capabilities">
          <span><i aria-hidden="true">✦</i> Adaptive insights</span>
          <span><i aria-hidden="true">↗</i> Model performance</span>
          <span><i aria-hidden="true">◈</i> Secure experiments</span>
        </div>
        <div className="login-security-note">
          <span aria-hidden="true" />
          <span>Private administrator access</span>
        </div>
      </section>
      <section className="login-panel" aria-label="Administrator sign in">
        <form className="login-card" onSubmit={(event) => void handleSubmit(event)}>
          <div className="login-card-topline">
            <span className="login-live-dot" aria-hidden="true" />
            <span>Secure workspace</span>
            <span className="login-card-lock" aria-hidden="true">⌁</span>
          </div>
          <div className="login-card-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 3 19 6v5c0 4.7-2.9 8-7 10-4.1-2-7-5.3-7-10V6l7-3Z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          </div>
          <p className="login-card-eyebrow">ADMINISTRATOR ACCESS</p>
          <h2>Welcome back</h2>
          <p className="login-card-copy">Sign in to open your intrusion detection dashboard.</p>
          <label htmlFor="admin-password">Administrator password</label>
          <div className="login-password-wrap">
            <input
              id="admin-password"
              className="login-password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              autoFocus
            />
            <button
              className="login-password-toggle"
              type="button"
              onClick={() => setShowPassword((visible) => !visible)}
              aria-label={showPassword ? "Hide password" : "Show password"}
              aria-pressed={showPassword}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {error && <p className="login-error" role="alert">{error}</p>}
          <button className="login-submit" type="submit" disabled={submitting || !password}>
            {submitting ? "Signing in..." : "Sign in"}
            <span aria-hidden="true">→</span>
          </button>
          <p className="login-card-footnote">
            <span aria-hidden="true">🔒</span> Protected access for authorized administrators
          </p>
        </form>
      </section>
    </main>
  );
}

function App() {
  const [authenticated, setAuthenticated] = useState(hasAdminSession);
  const [loginNotice, setLoginNotice] = useState("");

  useEffect(() => {
    const handleExpired = () => {
      logout();
      setAuthenticated(false);
      setLoginNotice("Your administrator session expired. Please sign in again.");
    };
    window.addEventListener("ids-auth-expired", handleExpired);
    return () => window.removeEventListener("ids-auth-expired", handleExpired);
  }, []);

  async function handleLogin(password: string) {
    await login(password);
    setLoginNotice("");
    setAuthenticated(true);
  }

  function handleLogout() {
    logout();
    setLoginNotice("You have been signed out.");
    setAuthenticated(false);
  }

  if (!authenticated) {
    return <LoginScreen notice={loginNotice} onLogin={handleLogin} />;
  }
  return <AppLayout onLogout={handleLogout} />;
}

export default App;
