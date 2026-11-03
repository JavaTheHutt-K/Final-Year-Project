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

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL
  ?? (import.meta.env.DEV
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "");

const api = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem("ids-admin-token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (
      error.response?.status === 401 &&
      !error.config?.url?.includes("/api/auth/login")
    ) {
      sessionStorage.removeItem("ids-admin-token");
      window.dispatchEvent(new Event("ids-auth-expired"));
    }
    return Promise.reject(error);
  },
);

export async function login(password: string): Promise<void> {
  const { data } = await api.post<{ access_token: string }>(
    "/api/auth/login",
    { password },
  );
  sessionStorage.setItem("ids-admin-token", data.access_token);
}

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

export async function getAIComparison(
  testSet: string,
): Promise<{ test_set: string; analysis: string }> {
  const { data } = await api.post<{ test_set: string; analysis: string }>(
    "/api/ai/compare",
    { test_set: testSet },
    { timeout: 45_000 },
  );
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

export interface AiInsightsResponse {
  run_id: string;
  model: string;
  insights: string;
}

export async function generateAiInsights(): Promise<AiInsightsResponse> {
  const { data } = await api.post<AiInsightsResponse>("/api/insights");
  return data;
}

