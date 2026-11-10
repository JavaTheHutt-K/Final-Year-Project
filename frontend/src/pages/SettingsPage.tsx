import { useEffect, useState, type FormEvent } from "react";
import { getApiErrorMessage, saveConfig } from "../api";
import { useDashboard } from "../DashboardContext";
import { PageHeading, Panel } from "../components";
import type { EditableExperimentSettings, ExperimentConfig } from "../types";

function editableSettings(config: ExperimentConfig): EditableExperimentSettings {
  return {
    dataset: {
      sample_fraction: config.dataset?.sample_fraction ?? 0.1,
    },
    preprocessing: {
      drop_nan_rows: config.preprocessing?.drop_nan_rows ?? true,
      drop_duplicate_rows: config.preprocessing?.drop_duplicate_rows ?? true,
      drop_constant_columns: config.preprocessing?.drop_constant_columns ?? true,
    },
    split: {
      initial_train_fraction: config.split?.initial_train_fraction ?? 0.7,
      update_fraction: config.split?.update_fraction ?? 0.1,
      test_fraction: config.split?.test_fraction ?? 0.2,
    },
    training: {
      random_seed: config.training?.random_seed ?? 42,
      clients: config.training?.clients ?? 3,
      initial_rounds: config.training?.initial_rounds ?? 5,
      update_rounds: config.training?.update_rounds ?? 2,
      local_epochs: config.training?.local_epochs ?? 1,
      partition: config.training?.partition === "class_skewed" ? "class_skewed" : "iid",
      participation_fraction: config.training?.participation_fraction ?? 1,
      min_participating_clients: config.training?.min_participating_clients ?? config.training?.clients ?? 3,
    },
  };
}

function SettingsPage() {
  const {
    config,
    loading,
    running,
    theme,
    setTheme,
    autoRefresh,
    setAutoRefresh,
    refreshInterval,
    setRefreshInterval,
    refreshDashboard,
  } = useDashboard();
  const [draft, setDraft] = useState<EditableExperimentSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (config && !draft) setDraft(editableSettings(config));
  }, [config, draft]);

  const splitTotal = draft
    ? draft.split.initial_train_fraction
      + draft.split.update_fraction
      + draft.split.test_fraction
    : 1;
  const validSplit = Math.abs(splitTotal - 1) < 0.001;

  const updateSplitPercent = (
    key: keyof EditableExperimentSettings["split"],
    value: number,
  ) => {
    setDraft((current) => current && ({
      ...current,
      split: { ...current.split, [key]: value / 100 },
    }));
    setSaved(false);
  };

  const handleSave = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!draft || !validSplit) return;
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    try {
      const updatedConfig = await saveConfig(draft);
      setDraft(editableSettings(updatedConfig));
      await refreshDashboard();
      setSaved(true);
    } catch (error) {
      setSaveError(getApiErrorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <PageHeading
        eyebrow="PREFERENCES"
        title="Settings"
        description="Adjust experiment behavior and personalize dashboard display."
      />
      <form className="settings-form" onSubmit={(event) => void handleSave(event)}>
        {!draft ? (
          <Panel title="Experiment settings">
            <div className="settings-feedback" role={loading ? "status" : "alert"}>
              {loading ? "Loading settings…" : "Settings could not be loaded. Check the API connection and refresh."}
            </div>
          </Panel>
        ) : (
          <>
            <div className="settings-grid">
              <Panel title="Dataset sampling" subtitle="Controls how much of the prepared dataset is used in the next run.">
                <label className="setting-control">
                  <span>Sample fraction</span>
                  <span className="setting-input-with-unit">
                    <input
                      type="number"
                      min="1"
                      max="100"
                      step="1"
                      value={Math.round(draft.dataset.sample_fraction * 100)}
                      onChange={(event) => {
                        setDraft((current) => current && ({
                          ...current,
                          dataset: { sample_fraction: Number(event.target.value) / 100 },
                        }));
                        setSaved(false);
                      }}
                      required
                    />
                    <span>%</span>
                  </span>
                </label>
              </Panel>

              <Panel title="Preprocessing" subtitle="Choose cleanup steps to apply before training.">
                <div className="settings-toggles">
                  {([
                    ["drop_nan_rows", "Remove rows with missing values"],
                    ["drop_duplicate_rows", "Remove duplicate rows"],
                    ["drop_constant_columns", "Remove constant columns"],
                  ] as const).map(([key, label]) => (
                    <label className="settings-toggle" key={key}>
                      <span>{label}</span>
                      <input
                        type="checkbox"
                        checked={draft.preprocessing[key]}
                        onChange={(event) => {
                          setDraft((current) => current && ({
                            ...current,
                            preprocessing: {
                              ...current.preprocessing,
                              [key]: event.target.checked,
                            },
                          }));
                          setSaved(false);
                        }}
                      />
                    </label>
                  ))}
                </div>
              </Panel>

              <Panel title="Data split" subtitle="Partition the prepared sample; percentages must total 100%.">
                <div className="settings-control-grid">
                  {([
                    ["initial_train_fraction", "Initial training"],
                    ["update_fraction", "Update data"],
                    ["test_fraction", "Test data"],
                  ] as const).map(([key, label]) => (
                    <label className="setting-control" key={key}>
                      <span>{label}</span>
                      <span className="setting-input-with-unit">
                        <input
                          type="number"
                          min="1"
                          max="98"
                          step="1"
                          value={Math.round(draft.split[key] * 100)}
                          onChange={(event) => updateSplitPercent(key, Number(event.target.value))}
                          required
                        />
                        <span>%</span>
                      </span>
                    </label>
                  ))}
                </div>
                <p className={`settings-validation ${validSplit ? "" : "invalid"}`}>
                  Total: {Math.round(splitTotal * 100)}% {validSplit ? "· Valid split" : "· Must equal 100%"}
                </p>
              </Panel>

              <Panel title="Training" subtitle="Set federated client count and training workload.">
                <div className="settings-control-grid">
                  <label className="setting-control">
                    <span>Clients</span>
                    <input type="number" min="1" max="20" step="1" value={draft.training.clients} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, clients: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Initial rounds</span>
                    <input type="number" min="1" max="100" step="1" value={draft.training.initial_rounds} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, initial_rounds: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Update rounds</span>
                    <input type="number" min="1" max="100" step="1" value={draft.training.update_rounds} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, update_rounds: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Local epochs</span>
                    <input type="number" min="1" max="20" step="1" value={draft.training.local_epochs} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, local_epochs: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Participation fraction</span>
                    <span className="setting-input-with-unit">
                      <input type="number" min="1" max="100" step="1" value={Math.round(draft.training.participation_fraction * 100)} onChange={(event) => {
                        setDraft((current) => current && ({ ...current, training: { ...current.training, participation_fraction: Number(event.target.value) / 100 } }));
                        setSaved(false);
                      }} required />
                      <span>%</span>
                    </span>
                  </label>
                  <label className="setting-control">
                    <span>Minimum participants</span>
                    <input type="number" min="1" max={draft.training.clients} step="1" value={draft.training.min_participating_clients} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, min_participating_clients: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Random seed</span>
                    <input type="number" min="0" max="4294967295" step="1" value={draft.training.random_seed} onChange={(event) => {
                      setDraft((current) => current && ({ ...current, training: { ...current.training, random_seed: Number(event.target.value) } }));
                      setSaved(false);
                    }} required />
                  </label>
                  <label className="setting-control">
                    <span>Client partition</span>
                    <select value={draft.training.partition} onChange={(event) => {
                      const partition = event.target.value as EditableExperimentSettings["training"]["partition"];
                      setDraft((current) => current && ({ ...current, training: { ...current.training, partition } }));
                      setSaved(false);
                    }}>
                      <option value="iid">IID (balanced)</option>
                      <option value="class_skewed">Class-skewed</option>
                    </select>
                  </label>
                </div>
              </Panel>
            </div>
            <div className="settings-save-row">
              <span className="settings-feedback" role={saveError ? "alert" : "status"}>
                {saveError ?? (saved ? "Experiment settings saved for the next run." : running ? "Stop the active run before saving changes." : "")}
              </span>
              <button className="button button-primary" type="submit" disabled={saving || loading || running || !validSplit}>
                {saving ? "Saving…" : "Save experiment settings"}
              </button>
            </div>
          </>
        )}
      </form>

      <Panel title="Dashboard preferences" subtitle="These display choices are saved in this browser.">
        <div className="settings-preferences">
          <label className="setting-control">
            <span>Color theme</span>
            <select value={theme} onChange={(event) => setTheme(event.target.value as "light" | "dark")}>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <label className="settings-toggle">
            <span>Automatically refresh dashboard data</span>
            <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
          </label>
          <label className="setting-control">
            <span>Refresh interval</span>
            <select value={refreshInterval} onChange={(event) => setRefreshInterval(Number(event.target.value))} disabled={!autoRefresh}>
              <option value={15}>Every 15 seconds</option>
              <option value={30}>Every 30 seconds</option>
              <option value={60}>Every minute</option>
            </select>
          </label>
        </div>
      </Panel>
    </>
  );
}

export default SettingsPage;
