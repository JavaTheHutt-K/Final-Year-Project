import { Link } from "react-router-dom";
import { useDashboard } from "../DashboardContext";
import { EmptyState, MetricCard, PageHeading, Panel } from "../components";
import type { ClientRoundDetail, ClientStatus, TrainingHistoryEntry } from "../types";

function DevicesPage() {
  const { config, history, results, loading } = useDashboard();
  const federatedHistory = (history.length ? history : results?.training_history ?? [])
    .filter((entry) => !entry.model || entry.model === "federated");
  const legacyClientIds = new Set(
    federatedHistory.flatMap((entry) => Object.keys(entry.clients ?? {}).map(normalizeClientId)),
  );
  const clientCount = Math.max(config?.training?.clients ?? 0, legacyClientIds.size);
  const latestRound = federatedHistory.reduce(
    (latest, entry) => entry.round > (latest?.round ?? -1) ? entry : latest,
    undefined as (typeof federatedHistory)[number] | undefined,
  );
  const latestDetails = new Map<string, ClientRoundDetail>();
  for (const entry of federatedHistory) {
    for (const detail of entry.client_details ?? []) latestDetails.set(detail.client_id, detail);
    for (const [rawClientId, samples] of Object.entries(entry.clients ?? {})) {
      const clientId = normalizeClientId(rawClientId);
      if (!latestDetails.has(clientId)) {
        latestDetails.set(clientId, {
          client_id: clientId,
          partition_id: "recorded partition",
          status: "completed",
          availability: "available",
          selected: true,
          samples,
        });
      }
    }
  }
  const clients = Array.from({ length: clientCount }, (_, index) => {
    const clientId = `client-${index + 1}`;
    return latestDetails.get(clientId) ?? {
      client_id: clientId,
      partition_id: `partition-${index + 1}`,
      status: "registered" as ClientStatus,
      availability: "unknown" as const,
      selected: false,
      samples: 0,
    };
  });
  const selectedCount = latestRound?.selected_clients?.length ?? 0;
  const successfulCount = latestRound?.successful_participants?.length ?? 0;
  const failedCount = latestRound?.failed_participants?.length ?? 0;

  return (
    <>
      <PageHeading
        eyebrow="FEDERATED CLIENTS"
        title="Devices"
        description="Monitor simulated Flower clients and their assigned partitions. These are not physical or network devices."
      />
      <div className="metric-grid device-metrics">
        <MetricCard label="CONFIGURED CLIENTS" value={clientCount ? clientCount.toLocaleString() : "—"} note="Registered simulation clients" accent="#2587f4" icon="▣" />
        <MetricCard label="ROUNDS RECORDED" value={federatedHistory.length ? federatedHistory.length.toLocaleString() : "—"} note="Federated training rounds" accent="#20ad73" icon="↻" />
        <MetricCard label="LATEST ROUND" value={latestRound ? String(latestRound.round) : "—"} note="Most recent client contribution" accent="#8755ea" icon="◷" />
        <MetricCard label="ROUND PARTICIPATION" value={latestRound ? `${successfulCount}/${selectedCount}` : "—"} note={`${failedCount} failed · simulation only`} accent="#e5a13c" icon="!" />
      </div>
      <Panel
        title="Federated client participants"
        subtitle="Stable IDs, assigned partitions, and measured local evaluation"
        action={<Link className="panel-caption" to="/settings">Configure future runs →</Link>}
      >
        {clients.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Client</th><th>Partition</th><th>Samples</th><th>Local loss</th><th>Local F1</th><th>Status</th></tr></thead>
              <tbody>{clients.map((client) => (
                <tr key={client.client_id}>
                  <td>{client.client_id}</td>
                  <td>{client.partition_id}</td>
                  <td>{client.samples ? client.samples.toLocaleString() : "—"}</td>
                  <td>{formatNumber(client.local_training_loss)}</td>
                  <td>{formatNumber(client.f1_score, 3)}</td>
                  <td><span className={`participant-status participant-${client.status}`}>{client.status}</span></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <EmptyState icon="▣" message={loading ? "Loading federated client configuration…" : "No federated clients are configured."} />}
      </Panel>
      <div className="page-grid">
        <Panel title="Client status cards" subtitle="Availability is observed only when Flower returns an update; registration is not online status.">
          <div className="client-card-grid">{clients.map((client) => <ClientCard key={client.client_id} client={client} round={latestRound?.round} />)}</div>
        </Panel>
        <Panel title="Round timeline" subtitle="Aggregate visibility for the selected experiment run">
          {federatedHistory.length ? (
            <div className="client-timeline">{federatedHistory.map((entry) => <TimelineEntry key={`${entry.model ?? "federated"}-${entry.round}`} entry={entry} />)}</div>
          ) : <EmptyState message={loading ? "Loading round activity…" : "Round activity will appear when federated training runs."} />}
        </Panel>
      </div>
      <Panel title="Per-client participation history" subtitle="Measured updates returned by the Flower simulation">
        {clients.some((client) => latestDetails.has(client.client_id)) ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Round</th><th>Client</th><th>Status</th><th>Training duration</th><th>Evaluation duration</th><th>Recall</th><th>Error</th></tr></thead>
              <tbody>{federatedHistory.flatMap((entry) => (entry.client_details ?? []).map((client) => (
                <tr key={`${entry.model ?? "federated"}-${entry.round}-${client.client_id}`}>
                  <td>{entry.round}</td><td>{client.client_id}</td>
                  <td><span className={`participant-status participant-${client.status}`}>{client.status}</span></td>
                  <td>{formatDuration(client.training_duration_seconds)}</td><td>{formatDuration(client.evaluation_duration_seconds)}</td>
                  <td>{formatNumber(client.recall, 3)}</td><td>{client.error ?? "—"}</td>
                </tr>
              )))}</tbody>
            </table>
          </div>
        ) : <EmptyState message="No individual client updates have been recorded yet." />}
      </Panel>
    </>
  );
}

function normalizeClientId(clientId: string): string {
  const match = /^Client\s+(\d+)$/i.exec(clientId.trim());
  return match ? `client-${match[1]}` : clientId;
}

function formatNumber(value: number | undefined, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function formatDuration(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "—";
}

function ClientCard({ client, round }: { client: ClientRoundDetail; round?: number }) {
  return (
    <div className="client-card">
      <div className="client-card-heading"><strong>{client.client_id}</strong><span className={`participant-status participant-${client.status}`}>{client.status}</span></div>
      <small>{client.partition_id} · {client.availability} · {client.selected ? `round ${round ?? "—"}` : "not selected"}</small>
      <div className="client-card-metrics">
        <span><b>{client.samples.toLocaleString()}</b><small>samples</small></span>
        <span><b>{formatNumber(client.local_training_loss)}</b><small>train loss</small></span>
        <span><b>{formatNumber(client.f1_score, 3)}</b><small>local F1</small></span>
      </div>
      {client.error && <p className="status-error-text">{client.error}</p>}
      {client.last_activity && <small>Last activity: {new Date(client.last_activity).toLocaleString()}</small>}
    </div>
  );
}

function TimelineEntry({ entry }: { entry: TrainingHistoryEntry }) {
  const success = entry.successful_participants?.length ?? 0;
  const selected = entry.selected_clients?.length ?? 0;
  return (
    <div className="client-timeline-entry">
      <strong>Round {entry.round}</strong><span>{success}/{selected} successful</span>
      <small>{entry.aggregation_completed ? "Aggregation complete" : "Aggregation incomplete"}</small>
    </div>
  );
}

export default DevicesPage;
