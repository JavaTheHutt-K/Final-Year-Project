import { useDashboard } from "../DashboardContext";
import { EmptyState, PageHeading, Panel } from "../components";

const SETTING_SECTIONS = [
  { key: "dataset", title: "Dataset" },
  { key: "preprocessing", title: "Preprocessing" },
  { key: "split", title: "Data split" },
  { key: "model", title: "Model" },
  { key: "training", title: "Training" },
];

function SettingsPage() {
  const { config, loading } = useDashboard();

  return (
    <>
      <PageHeading
        eyebrow="CONFIGURATION"
        title="Settings"
        description="View the active dataset, preprocessing, model, and training settings."
      />
      {config ? (
        <div className="settings-grid">
          {SETTING_SECTIONS.map((section) => (
            <Panel title={section.title} key={section.key}>
              <pre className="settings-json">
                {JSON.stringify(config[section.key] ?? {}, null, 2)}
              </pre>
            </Panel>
          ))}
        </div>
      ) : (
        <Panel title="Experiment settings">
          <EmptyState message={loading ? "Loading active configuration…" : "Settings are not available until the API is connected."} />
        </Panel>
      )}
    </>
  );
}

export default SettingsPage;
