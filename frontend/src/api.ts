import axios, { type AxiosError, type InternalAxiosRequestConfig } from "axios";
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

const ADMIN_TOKEN_KEY = "ids-admin-token";
const AUTH_EXPIRED_EVENT = "ids-auth-expired";

interface LoginResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
}

function isProtectedRequest(config: InternalAxiosRequestConfig): boolean {
  const method = config.method?.toUpperCase();
  const path = new URL(config.url ?? "", config.baseURL || window.location.origin).pathname;
  return (
    (method === "PUT" && path === "/api/config") ||
    (method === "POST" &&
      ["/api/run", "/api/ai/compare", "/api/insights"].includes(path))
  );
}

function expireAdminSession(): void {
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
  window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
}

api.interceptors.request.use((config) => {
  if (!isProtectedRequest(config)) return config;
  const token = sessionStorage.getItem(ADMIN_TOKEN_KEY);
  if (!token) {
    expireAdminSession();
    return Promise.reject(new Error("Administrator login is required."));
  }
  config.headers.set("Authorization", `Bearer ${token}`);
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: string | Record<string, unknown>; message?: string }>) => {
    if (
      error.config &&
      isProtectedRequest(error.config as InternalAxiosRequestConfig) &&
      (error.response?.status === 401 || error.response?.status === 403)
    ) {
      expireAdminSession();
    }
    return Promise.reject(error);
  },
);

export async function login(password: string): Promise<void> {
  const { data } = await api.post<LoginResponse>("/api/auth/login", { password });
  if (
    typeof data.access_token !== "string" ||
    !data.access_token ||
    typeof data.token_type !== "string" ||
    data.token_type.toLowerCase() !== "bearer"
  ) {
    throw new Error("The server returned an invalid administrator login response.");
  }
  sessionStorage.setItem(ADMIN_TOKEN_KEY, data.access_token);
}

export function hasAdminSession(): boolean {
  return Boolean(sessionStorage.getItem(ADMIN_TOKEN_KEY));
}

export function logout(): void {
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
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
): Promise<{ test_set: string; analysis: string; source: "groq" | "deterministic-fallback"; model: string }> {
  const { data } = await api.post<{
    test_set: string;
    analysis: string;
    source: "groq" | "deterministic-fallback";
    model: string;
  }>(
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
  if (axios.isAxiosError<{ detail?: unknown; message?: unknown }>(error)) {
    if (!error.response) return "Could not reach the backend. Check your connection and try again.";
    const detail = error.response.data?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") {
            return item.msg;
          }
          return "";
        })
        .filter(Boolean);
      if (messages.length) return messages.join("; ");
    }
    if (typeof error.response.data?.message === "string") return error.response.data.message;
    if (error.response.status === 401) return "Your administrator session is invalid or has expired. Please sign in again.";
    if (error.response.status === 403) return "You are not authorized to perform this action.";
    return error.message || `The backend returned HTTP ${error.response.status}.`;
  }
  if (error instanceof SyntaxError) return "The backend returned an invalid response.";
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

export interface AiInsightsResponse {
  run_id: string;
  model: string;
  source: "groq" | "deterministic-fallback";
  insights: string;
}

export async function generateAiInsights(question?: string): Promise<AiInsightsResponse> {
  const { data } = await api.post<AiInsightsResponse>(
    "/api/insights",
    question?.trim() ? { question: question.trim() } : {},
    { timeout: 45_000 },
  );
  return data;
}
