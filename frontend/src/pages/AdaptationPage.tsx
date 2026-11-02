import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MODEL_COLORS, MODEL_IDS, MODEL_LABELS } from "../dashboardConstants";
import { useDashboard } from "../DashboardContext";
import { EmptyState, PageHeading, Panel } from "../components";

function AdaptationPage() {
  const { results, testSets, loading } = useDashboard();
  const testSet = testSets[0];
  const chartData = MODEL_IDS.map((model) => ({
    model: MODEL_LABELS[model],
    modelId: model,
    f1_score: results?.metrics?.[model]?.[testSet]?.f1_score,
    recall: results?.metrics?.[model]?.[testSet]?.recall,
  }));
  const hasMetrics = chartData.some((entry) => typeof entry.f1_score === "number");

  return (
    <>
      <PageHeading
        eyebrow="MODEL PERFORMANCE"
        title="Update comparison"
        description="Compare baseline and update-trained models on the same random holdout set."
      />
      <div className="trend-summary">
        <span className="trend-summary-icon">↗</span>
        <div>
          <strong>{testSet ?? "Holdout results"}</strong>
          <small>The update partition is random, not newer or time-ordered traffic.</small>
        </div>
        <span className="trend-summary-days">{testSets.length || "—"} test sets</span>
      </div>
      <div className="page-grid">
        <Panel title="F1-score by model" subtitle="All variants evaluated on the same held-out records">
          {hasMetrics ? (
            <MetricChart chartData={chartData} metric="f1_score" />
          ) : <EmptyState message={loading ? "Loading holdout metrics…" : "Run evaluation to compare model F1-scores."} />}
        </Panel>
        <Panel title="Recall by model" subtitle="Malicious traffic detection sensitivity on the holdout">
          {hasMetrics ? (
            <MetricChart chartData={chartData} metric="recall" />
          ) : <EmptyState message={loading ? "Loading holdout metrics…" : "Recall results will appear after evaluation."} />}
        </Panel>
      </div>
      <Panel title="Model and update key" subtitle="Colors identify the four measured model variants">
        <div className="model-legend-grid">
          {MODEL_IDS.map((model) => (
            <div className="model-legend-item" key={model}>
              <span style={{ background: MODEL_COLORS[model] }} />
              <strong>{MODEL_LABELS[model]}</strong>
              <small>{model.includes("updated") ? "Trained further with the update partition" : "Initial model"}</small>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

function MetricChart({
  chartData,
  metric,
}: {
  chartData: {
    model: string;
    modelId: (typeof MODEL_IDS)[number];
    f1_score?: number;
    recall?: number;
  }[];
  metric: "f1_score" | "recall";
}) {
  return (
    <div className="chart-wrap chart-xl">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 13, right: 18, left: -16, bottom: 0 }}>
          <CartesianGrid stroke="#edf1f6" vertical={false} />
          <XAxis dataKey="model" axisLine={false} tickLine={false} interval={0} tick={{ fill: "#8290a2", fontSize: 10 }} />
          <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
          <Tooltip formatter={(value: number) => value.toFixed(4)} />
          <Bar dataKey={metric} name={metric === "f1_score" ? "F1-score" : "Recall"} radius={[5, 5, 0, 0]}>
            {chartData.map((entry) => (
              <Cell key={entry.modelId} fill={MODEL_COLORS[entry.modelId]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default AdaptationPage;
