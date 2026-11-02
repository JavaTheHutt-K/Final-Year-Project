import { createContext, useContext } from "react";
import type {
  DayMetrics,
  ExperimentConfig,
  ExperimentResults,
  ModelId,
  RunStatus,
  TrainingHistoryEntry,
} from "./types";

export interface MetricRow {
  model: ModelId;
  testSet: string;
  metrics: DayMetrics;
}

export interface DashboardContextValue {
  config: ExperimentConfig | null;
  results: ExperimentResults | null;
  history: TrainingHistoryEntry[];
  status: RunStatus;
  loading: boolean;
  running: boolean;
  error: string | null;
  testSets: string[];
  selectedTestSet: string;
  setSelectedTestSet: (testSet: string) => void;
  theme: "light" | "dark";
  setTheme: (theme: "light" | "dark") => void;
  autoRefresh: boolean;
  setAutoRefresh: (enabled: boolean) => void;
  refreshInterval: number;
  setRefreshInterval: (seconds: number) => void;
  refreshDashboard: () => Promise<void>;
  refreshStatus: () => Promise<void>;
}

export const DashboardContext = createContext<DashboardContextValue | null>(null);

export function useDashboard(): DashboardContextValue {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used within the dashboard layout");
  }
  return context;
}

export function metricRowsFor(
  results: ExperimentResults | null,
  models: ModelId[],
): MetricRow[] {
  const rows: MetricRow[] = [];
  for (const model of models) {
    for (const [testSet, metrics] of Object.entries(results?.metrics?.[model] ?? {})) {
      rows.push({ model, testSet, metrics });
    }
  }
  return rows;
}
