import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useDashboard } from "../DashboardContext";
import { EmptyState, PageHeading, Panel } from "../components";
import type { TrainingHistoryEntry } from "../types";

const METRIC_LINES = [
  { key: "precision", name: "Precision", color: "#3878ee" },
  { key: "recall", name: "Recall", color: "#16a47a" },
  { key: "f1_score", name: "F1-score", color: "#8d67db" },
  { key: "false_positive_rate", name: "False-positive rate", color: "#e5a13c" },
] as const;

function TrainingPage() {
  const { history, results, loading } = useDashboard();
  const [showClients, setShowClients] = useState(true);
  const allHistory = history.length ? history : results?.training_history ?? [];
  const federatedHistory = useMemo(
    () => allHistory.filter((item) => !item.model || item.model === "federated"),
    [allHistory],
  );
  const clientNames = [
    ...new Set(federatedHistory.flatMap((item) => Object.keys(item.clients ?? {}))),
  ];

  return (
    <>
      <PageHeading
        eyebrow="TRAINING"
        title="Federated training"
        description="Follow model metrics after each federated training round and inspect available client contributions."
        action={clientNames.length > 0 ? (
          <button className={`toggle-button ${showClients ? "selected" : ""}`} onClick={() => setShowClients((shown) => !shown)}>
            <span className="toggle-dot" /> Client contributions
          </button>
        ) : undefined}
      />
      <div className="training-summary">
        <div><span className="summary-icon blue">⌁</span><span><small>ROUNDS RECORDED</small><strong>{federatedHistory.length || "—"}</strong></span></div>
        <div><span className="summary-icon green">◎</span><span><small>CLIENTS TRACKED</small><strong>{clientNames.length || "—"}</strong></span></div>
        <div><span className="summary-icon purple">◷</span><span><small>LATEST ROUND</small><strong>{federatedHistory[federatedHistory.length - 1]?.round ?? "—"}</strong></span></div>
      </div>
      <Panel title="Model metrics per round" subtitle="Federated model · precision, recall, F1-score and false-positive rate">
        {federatedHistory.length ? (
          <div className="chart-wrap chart-xl">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={federatedHistory} margin={{ top: 15, right: 24, left: -15, bottom: 0 }}>
                <CartesianGrid stroke="#edf1f6" vertical={false} />
                <XAxis dataKey="round" tickFormatter={(value: number) => `Round ${value}`} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
                <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
                <Tooltip formatter={(value: number) => value.toFixed(4)} labelFormatter={(value) => `Round ${value}`} />
                <Legend iconType="circle" />
                {METRIC_LINES.map((metric) => (
                  <Line key={metric.key} type="monotone" dataKey={metric.key} name={metric.name} stroke={metric.color} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState message={loading ? "Loading training history…" : "Training history will appear after an experiment has run."} />}
      </Panel>
      {showClients && clientNames.length > 0 && (
        <Panel title="Client contributions" subtitle="Contribution values recorded for each round" className="client-panel">
          <div className="chart-wrap chart-large">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={federatedHistory} margin={{ top: 12, right: 18, left: -14, bottom: 0 }}>
                <CartesianGrid stroke="#edf1f6" vertical={false} />
                <XAxis dataKey="round" tickFormatter={(value: number) => `Round ${value}`} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
                <Tooltip labelFormatter={(value) => `Round ${value}`} />
                <Legend iconType="circle" />
                {clientNames.map((client, index) => (
                  <Line
                    key={client}
                    type="monotone"
                    dataKey={(entry: TrainingHistoryEntry) => entry.clients?.[client]}
                    name={client}
                    stroke={["#3878ee", "#16a47a", "#8d67db"][index % 3]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}
      {!clientNames.length && federatedHistory.length > 0 && (
        <div className="info-strip"><span>i</span> Client-specific contribution data is not included in the available training history.</div>
      )}
    </>
  );
}

export default TrainingPage;
