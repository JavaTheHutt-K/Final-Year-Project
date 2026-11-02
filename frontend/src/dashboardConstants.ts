import type { MetricName, ModelId } from "./types";

export const MODEL_IDS: ModelId[] = [
  "federated",
  "centralised",
  "federated_updated",
  "centralised_updated",
];

export const MODEL_LABELS: Record<ModelId, string> = {
  federated: "Federated",
  centralised: "Centralised",
  federated_updated: "Federated · updated",
  centralised_updated: "Centralised · updated",
};

export const MODEL_COLORS: Record<ModelId, string> = {
  federated: "#3878ee",
  centralised: "#16a47a",
  federated_updated: "#8d67db",
  centralised_updated: "#e5a13c",
};

export const METRICS: { value: MetricName; label: string }[] = [
  { value: "precision", label: "Precision" },
  { value: "recall", label: "Recall" },
  { value: "f1_score", label: "F1-score" },
  { value: "false_positive_rate", label: "False-positive rate" },
];

export function formatMetric(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "—";
}
