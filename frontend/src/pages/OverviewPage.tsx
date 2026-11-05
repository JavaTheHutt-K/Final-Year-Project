import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getApiErrorMessage, startExperiment } from "../api";
import { useDashboard, metricRowsFor } from "../DashboardContext";
import { MODEL_COLORS, MODEL_IDS, MODEL_LABELS, METRICS, formatModelAxisLabel } from "../dashboardConstants";
import { EmptyState, MetricCard, Panel } from "../components";

const ATTACK_COLORS = ["#f05252", "#ff9f43", "#f7c948", "#3588f5", "#8d62e8", "#20b98b", "#94a3b8"];

function OverviewPage() {
  const {
    config,
    results,
    history,
    loading,
    running,
    selectedTestSet,
    status,
    refreshStatus,
  } = useDashboard();
  const [runError, setRunError] = useState<string | null>(null);
  const rows = useMemo(() => metricRowsFor(results, MODEL_IDS), [results]);
  const allTestSets = Object.keys(results?.metrics?.federated ?? {});
  const visibleTestSets = selectedTestSet ? [selectedTestSet] : allTestSets;
  const federatedMetrics = visibleTestSets
    .map((testSet) => results?.metrics?.federated?.[testSet])
    .filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
  const aggregate = federatedMetrics.reduce(
    (sum, entry) => ({
      total: sum.total + entry.confusion_matrix.tn + entry.confusion_matrix.fp + entry.confusion_matrix.fn + entry.confusion_matrix.tp,
      detectedIntrusions: sum.detectedIntrusions + entry.confusion_matrix.tp,
      normalTraffic: sum.normalTraffic + entry.confusion_matrix.tn + entry.confusion_matrix.fp,
    }),
    { total: 0, detectedIntrusions: 0, normalTraffic: 0 },
  );
  const trafficTotals = Object.values(results?.data_report?.sources ?? {}).reduce(
    (totals, report) => ({
      benign: totals.benign + (report.class_balance?.benign ?? 0),
      malicious: totals.malicious + (report.class_balance?.malicious ?? 0),
    }),
    { benign: 0, malicious: 0 },
  );
  const trafficData = [
    { traffic: "Normal", records: trafficTotals.benign, color: "#20b879" },
    { traffic: "Intrusion", records: trafficTotals.malicious, color: "#f05252" },
  ].filter((entry) => entry.records > 0);
  const benignLabel = config?.dataset?.benign_label?.trim().toUpperCase() ?? "";
  const attackCounts = Object.entries(results?.data_report?.sources ?? {})
    .reduce<Record<string, number>>((counts, [, source]) => {
      for (const [label, count] of Object.entries(source.attack_label_counts ?? {})) {
        if (label.trim().toUpperCase() !== benignLabel) {
          counts[label] = (counts[label] ?? 0) + count;
        }
      }
      return counts;
    }, {});
  const attackData = Object.entries(attackCounts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const detectionTrends = history
    .filter((entry) => !entry.model || entry.model === "federated")
    .map((entry) => ({
      round: entry.round,
      Recall: entry.recall,
      "False-positive rate": entry.false_positive_rate,
    }));
  const averagePerformance = (["centralised", "federated"] as const).map((model) => {
    const modelTestSets = visibleTestSets
      .map((testSet) => results?.metrics?.[model]?.[testSet])
      .filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
    const averages: Record<(typeof METRICS)[number]["value"], number | undefined> = {
      precision: undefined,
      recall: undefined,
      f1_score: undefined,
      false_positive_rate: undefined,
    };
    for (const { value } of METRICS) {
      averages[value] = modelTestSets.length
        ? modelTestSets.reduce((sum, metrics) => sum + metrics[value], 0) / modelTestSets.length
        : undefined;
    }
    return { model: MODEL_LABELS[model], modelId: model, ...averages };
  });
  const recentRows = rows
    .filter((row) => !selectedTestSet || row.testSet === selectedTestSet)
    .slice(0, 5);
  const latestRound = history
    .filter((entry) => !entry.model || entry.model === "federated")
    .reduce((latest, entry) => (entry.round > (latest?.round ?? -1) ? entry : latest), undefined as (typeof history)[number] | undefined);
  const clientRows = Object.entries(latestRound?.clients ?? {});

  const run = async () => {
    setRunError(null);
    try {
      await startExperiment();
      await refreshStatus();
    } catch (error) {
      setRunError(getApiErrorMessage(error));
    }
  };

  return (
    <div className="dashboard-page">
      <div className="metric-grid">
        <MetricCard label="TOTAL TRAFFIC" value={federatedMetrics.length ? aggregate.total.toLocaleString() : "—"} note={federatedMetrics.length ? "Records in selected test set" : "Awaiting evaluation"} accent="#2587f4" icon="⌘" />
        <MetricCard label="DETECTED INTRUSIONS" value={federatedMetrics.length ? aggregate.detectedIntrusions.toLocaleString() : "—"} note="Malicious records correctly detected" accent="#ef5350" icon="⬟" />
        <MetricCard label="NORMAL TRAFFIC" value={federatedMetrics.length ? aggregate.normalTraffic.toLocaleString() : "—"} note="Benign records in selected test set" accent="#20ad73" icon="✓" />
        <MetricCard label="ACTIVE CLIENTS" value={config?.training?.clients?.toLocaleString() ?? "—"} note="Simulated federated clients" accent="#8755ea" icon="▣" />
      </div>

      <div className="overview-primary-grid">
        <Panel title="Traffic class balance" subtitle="Class totals for the prepared traffic sample">
          {trafficData.length ? (
            <div className="chart-wrap chart-medium">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trafficData} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
                  <CartesianGrid stroke="#edf1f6" vertical={false} />
                  <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 9 }} tickFormatter={(value: number) => value.toLocaleString()} />
                  <YAxis type="category" dataKey="traffic" axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 9 }} width={65} />
                  <Tooltip formatter={(value: number) => value.toLocaleString()} />
                  <Bar dataKey="records" name="Records" radius={[0, 4, 4, 0]}>
                    {trafficData.map((entry) => <Cell key={entry.traffic} fill={entry.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message={loading ? "Loading traffic data…" : "Traffic counts will appear after data preparation."} />}
        </Panel>

        <Panel title="Attack types distribution" subtitle="Attack labels in the cleaned dataset">
          {attackData.length ? (
            <div className="attack-distribution">
              <div className="donut-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={attackData} dataKey="value" nameKey="name" innerRadius="57%" outerRadius="87%" paddingAngle={2} stroke="none">
                      {attackData.map((item, index) => <Cell key={item.name} fill={ATTACK_COLORS[index % ATTACK_COLORS.length]} />)}
                    </Pie>
                    <Tooltip formatter={(value: number) => value.toLocaleString()} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center"><strong>{attackData.reduce((sum, item) => sum + item.value, 0).toLocaleString()}</strong><small>intrusions</small></div>
              </div>
              <div className="attack-legend">
                {attackData.slice(0, 6).map((attack, index) => (
                  <div className="attack-legend-row" key={attack.name}>
                    <i style={{ background: ATTACK_COLORS[index % ATTACK_COLORS.length] }} />
                    <span title={attack.name}>{attack.name}</span>
                    <strong>{attack.value.toLocaleString()}</strong>
                  </div>
                ))}
              </div>
            </div>
          ) : <EmptyState message={loading ? "Loading attack labels…" : "Attack label counts are not available yet."} />}
        </Panel>
      </div>

      <div className="overview-primary-grid overview-secondary-grid">
        <Panel
          title="Detection results"
          subtitle="Model metrics by holdout test set"
          action={<span className="panel-caption">{recentRows.length} results</span>}
        >
          {recentRows.length ? (
            <div className="table-scroll">
              <table className="data-table compact-table">
                <thead><tr><th>Model</th><th>Test set</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr></thead>
                <tbody>{recentRows.map((row) => (
                  <tr key={`${row.model}-${row.testSet}`}>
                    <td><span className="model-key" style={{ background: MODEL_COLORS[row.model] }} />{MODEL_LABELS[row.model]}</td>
                    <td>{row.testSet}</td>
                    <td>{row.metrics.precision.toFixed(3)}</td>
                    <td>{row.metrics.recall.toFixed(3)}</td>
                    <td>{row.metrics.f1_score.toFixed(3)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <EmptyState message={loading ? "Loading model results…" : "Detection results will appear after an experiment."} />}
        </Panel>

        <Panel title="Detection performance over rounds" subtitle="Federated recall and false-positive rate">
          {detectionTrends.length ? (
            <div className="chart-wrap chart-small">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={detectionTrends} margin={{ top: 7, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="#edf1f6" vertical={false} />
                  <XAxis dataKey="round" tickFormatter={(value: number) => `Round ${value}`} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 8 }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 8 }} />
                  <Tooltip formatter={(value: number) => value.toFixed(4)} labelFormatter={(value) => `Round ${value}`} />
                  <Legend iconType="circle" />
                  <Line type="monotone" dataKey="Recall" stroke="#20b879" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="False-positive rate" stroke="#f05252" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message={loading ? "Loading detection trends…" : "Detection trends will appear after evaluation."} />}
        </Panel>
      </div>

      <div className="overview-primary-grid overview-secondary-grid">
        <Panel title="Model performance" subtitle="Baseline models · precision, recall and F1-score">
          {averagePerformance.some((entry) => typeof entry.f1_score === "number") ? (
            <div className="chart-wrap chart-medium">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={averagePerformance} margin={{ top: 8, right: 16, left: -20, bottom: 18 }}>
                  <CartesianGrid stroke="#edf1f6" vertical={false} />
                  <XAxis dataKey="model" axisLine={false} tickLine={false} tickFormatter={formatModelAxisLabel} angle={-25} textAnchor="end" height={54} tickMargin={7} tick={{ fill: "#718196", fontSize: 8 }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 8 }} />
                  <Tooltip formatter={(value: number) => value.toFixed(3)} />
                  <Legend iconType="circle" />
                  <Bar dataKey="precision" name="Precision" fill="#22b879" />
                  <Bar dataKey="recall" name="Recall" fill="#ff9f43" />
                  <Bar dataKey="f1_score" name="F1-score" fill="#8d62e8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message={loading ? "Loading performance metrics…" : "Model performance will appear after evaluation."} />}
        </Panel>

        <Panel title="Client model updates" subtitle={latestRound ? `Latest contribution data · round ${latestRound.round}` : "Federated client contributions"}>
          {clientRows.length ? (
            <div className="table-scroll">
              <table className="data-table compact-table">
                <thead><tr><th>Client ID</th><th>Round</th><th>Contribution</th></tr></thead>
                <tbody>{clientRows.map(([client, contribution], index) => (
                  <tr key={client}><td title={client}>Client {index + 1}</td><td>{latestRound?.round}</td><td>{typeof contribution === "number" ? contribution.toLocaleString() : "—"}</td></tr>
                ))}</tbody>
              </table>
            </div>
          ) : <EmptyState message={loading ? "Loading client updates…" : "Client contribution details are not included in the available training history."} />}
          <div className="run-status-inline">
            <span className={`status-pill status-${status.status}`}><i />{status.status === "queued" ? "Queued" : status.status === "running" ? "Training" : status.status === "completed" ? "Complete" : status.status === "failed" ? "Failed" : status.status === "unavailable" ? "Unavailable" : "Ready"}</span>
            <button className="button button-primary" disabled={running} onClick={() => void run()}>{status.status === "queued" ? "Queued" : status.status === "running" ? "Running" : "Run model"}</button>
            {runError && <small className="status-error-text">{runError}</small>}
          </div>
        </Panel>
      </div>
    </div>
  );
}

export default OverviewPage;
