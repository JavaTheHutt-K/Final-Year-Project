import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useDashboard } from "../DashboardContext";
import { EmptyState, MetricCard, PageHeading, Panel } from "../components";

function DatasetPage() {
  const { config, results, loading } = useDashboard();
  const report = results?.data_report;
  const sources = report?.sources ?? {};
  const sourceData = Object.entries(sources).map(([source, item]) => ({
    source,
    Benign: item.class_balance?.benign ?? 0,
    Malicious: item.class_balance?.malicious ?? 0,
  }));
  const partitions = Object.entries(report?.partitions ?? {});
  const attackLabels = Object.entries(sources).flatMap(([source, item]) =>
    Object.entries(item.attack_label_counts ?? {}).map(([label, count]) => ({
      source,
      label,
      count,
    })),
  );
  const sourceFileCount = config?.dataset?.source_file ? 1 : 0;

  return (
    <>
      <PageHeading
        eyebrow="DATASET"
        title="Traffic data"
        description="Review the cleaned lightweight dataset, class balance, and the reproducible training partitions."
      />
      <div className="metric-grid dataset-metrics">
        <MetricCard label="DATASET FILES" value={String(sourceFileCount || "—")} note={config?.dataset?.source_file ?? "Configured input"} accent="#2587f4" icon="◷" />
        <MetricCard label="ROWS AFTER CLEANING" value={report?.totals?.rows_after_cleaning?.toLocaleString() ?? "—"} note="Retained sample records" accent="#ef5350" icon="▤" />
        <MetricCard label="FEATURE COLUMNS" value={String(report?.totals?.feature_columns_after_cleaning ?? "—")} note="After preprocessing" accent="#20ad73" icon="▦" />
        <MetricCard label="CONSTANT COLUMNS" value={String(report?.totals?.constant_columns_dropped ?? report?.constant_columns_dropped?.length ?? "—")} note="Removed using initial training data" accent="#8755ea" icon="⊙" />
      </div>

      <div className="page-grid">
        <Panel title="Class balance" subtitle="Benign and malicious records after cleaning and sampling">
          {sourceData.length ? (
            <div className="chart-wrap chart-large">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sourceData} margin={{ top: 12, right: 16, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="#edf1f6" vertical={false} />
                  <XAxis dataKey="source" axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 11 }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: "#8290a2", fontSize: 11 }} />
                  <Tooltip />
                  <Legend iconType="circle" />
                  <Bar dataKey="Benign" fill="#3878ee" radius={[5, 5, 0, 0]} />
                  <Bar dataKey="Malicious" fill="#e5a13c" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState message={loading ? "Loading class balance…" : "Class balance will be shown after data preparation."} />}
        </Panel>

        <Panel title="Cleaning steps" subtitle="Source row counts and retained sample size">
          {Object.keys(sources).length ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead><tr><th>Source</th><th>Before</th><th>After NaNs</th><th>After duplicates</th><th>Sampled</th></tr></thead>
                <tbody>{Object.entries(sources).map(([source, item]) => (
                  <tr key={source}>
                    <td><span className="day-indicator" />{source}</td>
                    <td>{item.rows.before?.toLocaleString() ?? "—"}</td>
                    <td>{item.rows.after_nan_rows?.toLocaleString() ?? "—"}</td>
                    <td>{item.rows.after_duplicate_rows?.toLocaleString() ?? "—"}</td>
                    <td>{item.rows.after_sampling?.toLocaleString() ?? "—"}</td>
                  </tr>
                ))}</tbody>
              </table>
              <div className="table-footnote">
                Sample fraction: {report?.config?.sample_fraction ?? config?.dataset?.sample_fraction ?? "—"} ·
                Source rows: {report?.totals?.rows_before_cleaning?.toLocaleString() ?? "—"}
              </div>
            </div>
          ) : <EmptyState message={loading ? "Loading cleaning report…" : "The cleaning report will appear after data preparation."} />}
        </Panel>
      </div>

      <Panel title="Training partitions" subtitle="Reproducible stratified random split; not a time-ordered update">
        {partitions.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Partition</th><th>Rows</th><th>Benign</th><th>Malicious</th></tr></thead>
              <tbody>{partitions.map(([name, partition]) => (
                <tr key={name}>
                  <td>{name.replace(/_/g, " ")}</td>
                  <td>{partition.rows.toLocaleString()}</td>
                  <td>{partition.class_balance?.benign?.toLocaleString() ?? "—"}</td>
                  <td>{partition.class_balance?.malicious?.toLocaleString() ?? "—"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <EmptyState message={loading ? "Loading split details…" : "Partition counts will be shown after an experiment run."} />}
      </Panel>

      <Panel title="Attack label distribution" subtitle="Original labels retained for per-attack recall" className="dataset-attacks">
        {attackLabels.length ? (
          <div className="attack-label-list">
            {attackLabels.map(({ source, label, count }) => (
              <div className="attack-label-row" key={`${source}-${label}`}>
                <span>{label}</span><small>{source}</small>
                <div className="attack-bar"><i style={{ width: `${Math.min(100, (count / Math.max(1, report?.totals?.rows_after_cleaning ?? count)) * 100)}%` }} /></div>
                <strong>{count.toLocaleString()}</strong>
              </div>
            ))}
          </div>
        ) : <EmptyState message={loading ? "Loading label distribution…" : "Attack label counts will be shown when available."} />}
      </Panel>
    </>
  );
}

export default DatasetPage;
