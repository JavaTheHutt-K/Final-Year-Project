import { Link } from "react-router-dom";
import { useDashboard } from "../DashboardContext";
import { EmptyState, MetricCard, PageHeading, Panel } from "../components";

function DevicesPage() {
  const { config, history, results, loading } = useDashboard();
  const federatedHistory = (history.length ? history : results?.training_history ?? [])
    .filter((entry) => !entry.model || entry.model === "federated");
  const clientCount = config?.training?.clients ?? 0;
  const latestRound = federatedHistory.reduce(
    (latest, entry) => entry.round > (latest?.round ?? -1) ? entry : latest,
    undefined as (typeof federatedHistory)[number] | undefined,
  );
  const clients = Object.entries(latestRound?.clients ?? {});
  const clientRows = clients.length
    ? clients.map(([clientId, samples], index) => ({
        clientId,
        label: `Client ${index + 1}`,
        samples,
      }))
    : Array.from({ length: clientCount }, (_, index) => ({
        clientId: `configured-client-${index + 1}`,
        label: `Client ${index + 1}`,
        samples: undefined,
      }));

  return (
    <>
      <PageHeading
        eyebrow="FEDERATED CLIENTS"
        title="Devices"
        description="Inspect the simulated participants used by federated training. The dataset does not contain real device inventory."
      />
      <div className="metric-grid device-metrics">
        <MetricCard label="SIMULATED CLIENTS" value={clientCount ? clientCount.toLocaleString() : "—"} note="Configured for federated training" accent="#2587f4" icon="▣" />
        <MetricCard label="ROUNDS RECORDED" value={federatedHistory.length ? federatedHistory.length.toLocaleString() : "—"} note="Federated training rounds" accent="#20ad73" icon="↻" />
        <MetricCard label="LATEST ROUND" value={latestRound ? String(latestRound.round) : "—"} note="Most recent client contribution" accent="#8755ea" icon="◷" />
        <MetricCard label="DEVICE INVENTORY" value="Unavailable" note="Real device telemetry is not collected" accent="#e5a13c" icon="!" />
      </div>
      <Panel
        title="Federated client participants"
        subtitle="Simulated training participants · not physical or network devices"
        action={<Link className="panel-caption" to="/clients">View training details →</Link>}
      >
        {clientRows.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr><th>Participant</th><th>Type</th><th>Latest round</th><th>Training samples</th><th>Status</th></tr>
              </thead>
              <tbody>
                {clientRows.map((client) => (
                  <tr key={client.clientId}>
                    <td>{client.label}</td>
                    <td>Simulated Flower client</td>
                    <td>{latestRound?.round ?? "—"}</td>
                    <td>{typeof client.samples === "number" ? client.samples.toLocaleString() : "Not recorded"}</td>
                    <td><span className="participant-status">{latestRound ? "Recorded" : "Configured"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon="▣"
            message={loading ? "Loading federated client configuration…" : "No federated clients are configured. Set training.clients in config.yaml to list simulated participants."}
          />
        )}
      </Panel>
    </>
  );
}

export default DevicesPage;
