import { useCallback, useEffect, useMemo, useState } from "react";
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
  "/reports": "Reports",
  "/adaptation": "Model Performance",
  "/model-performance": "Model Performance",
  "/devices": "Devices",
  "/settings": "Settings",
};

function AppLayout() {
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
      setRunning(nextStatus.status === "running");
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

function App() {
  return <AppLayout />;
}

export default App;
