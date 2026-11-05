import { useMemo, useState } from "react";
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
import { MODEL_COLORS, MODEL_IDS, MODEL_LABELS, METRICS, formatMetric, formatModelAxisLabel } from "../dashboardConstants";
import { useDashboard, metricRowsFor } from "../DashboardContext";
import { EmptyState, PageHeading, Panel, SelectField } from "../components";
import type { MetricName, ModelId } from "../types";

function ResultsPage() {
  const { results, testSets, selectedTestSet, setSelectedTestSet, loading } = useDashboard();
  const [selectedMetric, setSelectedMetric] = useState<MetricName>("f1_score");
  const [selectedModel, setSelectedModel] = useState<ModelId>("federated");
  const activeTestSet = selectedTestSet || testSets[0] || "";
  const rows = useMemo(() => metricRowsFor(results, MODEL_IDS), [results]);
  const comparisonRows = rows.filter((row) => row.testSet === activeTestSet);
  const chartData = MODEL_IDS.map((model) => {
    const record = comparisonRows.find((row) => row.model === model);
    return {
      model: MODEL_LABELS[model],
      score: record?.metrics[selectedMetric] ?? 0,
      color: MODEL_COLORS[model],
      available: Boolean(record),
    };
  });
  const matrix = results?.metrics?.[selectedModel]?.[activeTestSet]?.confusion_matrix;
  const attackRecalls = results?.metrics?.[selectedModel]?.[activeTestSet]?.attack_type_recall;
  const exportCsv = () => {
    if (!rows.length) return;
    const header = ["model", "test_set", "precision", "recall", "f1_score", "false_positive_rate", "tn", "fp", "fn", "tp"];
    const csvRows = rows.map(({ model, testSet, metrics }) => [
      MODEL_LABELS[model],
      testSet,
      metrics.precision,
      metrics.recall,
      metrics.f1_score,
      metrics.false_positive_rate,
      metrics.confusion_matrix.tn,
      metrics.confusion_matrix.fp,
      metrics.confusion_matrix.fn,
      metrics.confusion_matrix.tp,
    ]);
    const csv = [header, ...csvRows]
      .map((row) => row.map((field) => `"${String(field).replace(/"/g, '""')}"`).join(","))
      .join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "intrusion-detection-results.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <PageHeading
        eyebrow="EVALUATION"
        title="Results comparison"
        description="Compare model metrics on the random holdout set and inspect confusion matrices."
        action={<button className="button button-outline" onClick={exportCsv} disabled={!rows.length}>↓ <span>Export CSV</span></button>}
      />
      <div className="filter-bar">
        <SelectField label="TEST SET" value={activeTestSet} options={testSets.map((testSet) => ({ value: testSet, label: testSet }))} onChange={setSelectedTestSet} />
        <SelectField label="METRIC" value={selectedMetric} options={METRICS.map((metric) => ({ value: metric.value, label: metric.label }))} onChange={(value) => setSelectedMetric(value as MetricName)} />
        <div className="neutral-note"><span>i</span> Malicious traffic is the positive class.</div>
      </div>
      <div className="page-grid results-top-grid">
        <Panel title={`${METRICS.find((metric) => metric.value === selectedMetric)?.label ?? "Metric"} by model`} subtitle={activeTestSet ? `Test set: ${activeTestSet}` : "Choose a test set"}>
          {comparisonRows.length ? (
            <div className="chart-wrap chart-large">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 14, right: 18, left: -16, bottom: 18 }}>
                  <CartesianGrid stroke="#edf1f6" vertical={false} />
                  <XAxis dataKey="model" axisLine={false} tickLine={false} interval={0} tickFormatter={formatModelAxisLabel} angle={-25} textAnchor="end" height={54} tickMargin={7} tick={{ fill: "#718196", fontSize: 9 }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 10 }} />
                  <Tooltip formatter={(value: number) => formatMetric(value)} />
                  <Bar dataKey="score" name={METRICS.find((metric) => metric.value === selectedMetric)?.label} radius={[5, 5, 0, 0]}>
                    {chartData.map((entry) => <Cell key={entry.model} fill={entry.color} opacity={entry.available ? 1 : 0.2} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message={loading ? "Loading evaluation metrics…" : "Evaluation metrics for this test set are not available."} />}
        </Panel>
        <Panel title="Confusion matrix" subtitle="Actual and predicted binary classes">
          <SelectField label="MODEL" value={selectedModel} options={MODEL_IDS.map((model) => ({ value: model, label: MODEL_LABELS[model] }))} onChange={(value) => setSelectedModel(value as ModelId)} />
          {matrix ? (
            <div className="matrix-layout">
              <span className="matrix-y-label">ACTUAL</span>
              <div className="matrix-content">
                <div className="matrix-x-label">PREDICTED</div>
                <div className="matrix-grid">
                  <MatrixCell kind="true" label="BENIGN" count={matrix.tn} description="True negative" />
                  <MatrixCell kind="error" label="MALICIOUS" count={matrix.fp} description="False positive" />
                  <MatrixCell kind="error" label="BENIGN" count={matrix.fn} description="False negative" />
                  <MatrixCell kind="true" label="MALICIOUS" count={matrix.tp} description="True positive" />
                </div>
                <div className="matrix-axis"><span>Benign</span><span>Malicious</span></div>
              </div>
            </div>
          ) : <EmptyState message={loading ? "Loading confusion matrix…" : "Choose a model and test set with evaluation data."} />}
        </Panel>
      </div>
      <Panel title="Metric details" subtitle={`All reported metrics · ${activeTestSet || "no test set selected"}`} className="results-table-panel">
        {comparisonRows.length ? (
          <div className="table-scroll">
            <table className="data-table results-table">
              <thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>False-positive rate</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr></thead>
              <tbody>{MODEL_IDS.map((model) => {
                const row = comparisonRows.find((item) => item.model === model);
                return <tr key={model}><td><span className="model-key" style={{ background: MODEL_COLORS[model] }} />{MODEL_LABELS[model]}</td><td>{formatMetric(row?.metrics.precision)}</td><td>{formatMetric(row?.metrics.recall)}</td><td>{formatMetric(row?.metrics.f1_score)}</td><td>{formatMetric(row?.metrics.false_positive_rate)}</td><td>{row?.metrics.confusion_matrix.tn.toLocaleString() ?? "—"}</td><td>{row?.metrics.confusion_matrix.fp.toLocaleString() ?? "—"}</td><td>{row?.metrics.confusion_matrix.fn.toLocaleString() ?? "—"}</td><td>{row?.metrics.confusion_matrix.tp.toLocaleString() ?? "—"}</td></tr>;
              })}</tbody>
            </table>
            <div className="table-footnote">False-positive rate = FP / (FP + TN).</div>
          </div>
        ) : <EmptyState message={loading ? "Loading results…" : "Results will appear after evaluation."} />}
      </Panel>
      {attackRecalls && Object.keys(attackRecalls).length > 0 && (
        <Panel title="Recall by attack type" subtitle={`${MODEL_LABELS[selectedModel]} · ${activeTestSet}`}>
          <div className="attack-label-list">{Object.entries(attackRecalls).map(([attack, recall]) => (
            <div className="attack-label-row" key={attack}>
              <span>{attack}</span><div className="attack-bar"><i style={{ width: `${Math.max(0, Math.min(100, recall * 100))}%` }} /></div><strong>{formatMetric(recall)}</strong>
            </div>
          ))}</div>
        </Panel>
      )}
    </>
  );
}

function MatrixCell({
  kind,
  label,
  count,
  description,
}: {
  kind: "true" | "error";
  label: string;
  count: number;
  description: string;
}) {
  return (
    <div className={`matrix-cell ${kind}`}>
      <small>{label}</small><strong>{count.toLocaleString()}</strong><span>{description}</span>
    </div>
  );
}

export default ResultsPage;
