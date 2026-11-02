import axios from "axios";
import type {
  ExperimentConfig,
  EditableExperimentSettings,
  ExperimentResults,
  MetricsResponse,
  ModelId,
  RunResponse,
  RunStatus,
  TrainingHistoryEntry,
} from "./types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

export async function getConfig(): Promise<ExperimentConfig> {
  const { data } = await api.get<ExperimentConfig>("/api/config");
  return data;
}

export async function saveConfig(
  settings: EditableExperimentSettings,
): Promise<ExperimentConfig> {
  const { data } = await api.put<ExperimentConfig>("/api/config", settings);
  return data;
}

export async function getResults(): Promise<ExperimentResults> {
  const { data } = await api.get<ExperimentResults>("/api/results");
  return data;
}

export async function getTrainingHistory(): Promise<TrainingHistoryEntry[]> {
  const { data } = await api.get<TrainingHistoryEntry[]>("/api/training-history");
  return data;
}

export async function getMetrics(
  model: ModelId,
  testSet: string,
): Promise<MetricsResponse> {
  const { data } = await api.get<MetricsResponse>("/api/metrics", {
    params: { model, test_set: testSet },
  });
  return data;
}

export async function startExperiment(): Promise<RunResponse> {
  const { data } = await api.post<RunResponse>("/api/run");
  return data;
}

export async function getRunStatus(): Promise<RunStatus> {
  const { data } = await api.get<RunStatus>("/api/run/status");
  return data;
}

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError<{ detail?: string; message?: string }>(error)) {
    return (
      error.response?.data?.detail ??
      error.response?.data?.message ??
      error.message
    );
  }
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}
