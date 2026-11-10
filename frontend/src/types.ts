export type ModelId =
  | "federated"
  | "centralised"
  | "federated_updated"
  | "centralised_updated";

export type MetricName =
  | "precision"
  | "recall"
  | "f1_score"
  | "false_positive_rate";

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface DayMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  false_positive_rate: number;
  confusion_matrix: ConfusionMatrix;
  attack_type_recall?: Record<string, number>;
}

export interface TrainingHistoryEntry {
  round: number;
  model?: ModelId;
  precision: number;
  recall: number;
  f1_score: number;
  false_positive_rate: number;
  clients?: Record<string, number>;
  expected_clients?: number;
  selected_clients?: string[];
  successful_participants?: string[];
  failed_participants?: string[];
  aggregation_completed?: boolean;
  global_evaluation?: {
    loss?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    false_positive_rate?: number;
  };
  client_details?: ClientRoundDetail[];
  run_id?: string;
}

export type ClientStatus = "registered" | "available" | "selected" | "training" | "completed" | "failed" | "unavailable";

export interface ClientRoundDetail {
  client_id: string;
  partition_id: string;
  status: ClientStatus;
  availability: "available" | "unavailable" | "unknown";
  selected: boolean;
  samples: number;
  local_training_loss?: number;
  training_duration_seconds?: number;
  loss?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  false_positive_rate?: number;
  evaluation_duration_seconds?: number;
  error?: string | null;
  last_activity?: string;
}

export interface CleaningSourceReport {
  file?: string;
  files?: string[];
  rows: Record<string, number>;
  class_balance?: {
    benign: number;
    malicious: number;
  };
  attack_label_counts?: Record<string, number>;
}

export interface DataReport {
  config?: {
    raw_data_dir?: string;
    label_column?: string;
    benign_label?: string;
    sample_fraction?: number;
  };
  sources: Record<string, CleaningSourceReport>;
  partitions?: Record<string, {
    rows: number;
    class_balance?: {
      benign: number;
      malicious: number;
    };
  }>;
  totals?: {
    rows_before_cleaning?: number;
    rows_after_cleaning?: number;
    columns_after_cleaning?: number;
    feature_columns_after_cleaning?: number;
    constant_columns_dropped?: number;
  };
  constant_columns_dropped?: string[];
}

export interface ExperimentConfig {
  project?: {
    title?: string;
    description?: string;
  };
  dataset?: {
    source_file?: string;
    chunk_size?: number;
    label_column?: string;
    benign_label?: string;
    attack_label_column?: string;
    binary_label_column?: string;
    sample_fraction?: number;
  };
  split?: {
    strategy?: string;
    initial_train_fraction?: number;
    update_fraction?: number;
    test_fraction?: number;
    test_set_name?: string;
  };
  preprocessing?: {
    drop_nan_rows?: boolean;
    drop_duplicate_rows?: boolean;
    drop_constant_columns?: boolean;
    scaler?: string;
  };
  model?: {
    type?: string;
    hyperparameters?: Record<string, number | string | boolean>;
  };
  training?: {
    clients?: number;
    initial_rounds?: number;
    update_rounds?: number;
    local_epochs?: number;
    partition?: string;
    aggregation?: string;
    random_seed?: number;
    participation_fraction?: number;
    min_participating_clients?: number;
  };
  paths?: Record<string, string>;
  api?: {
    host?: string;
    port?: number;
    cors_origins?: string[];
  };
  [section: string]: unknown;
}

export interface EditableExperimentSettings {
  dataset: {
    sample_fraction: number;
  };
  preprocessing: {
    drop_nan_rows: boolean;
    drop_duplicate_rows: boolean;
    drop_constant_columns: boolean;
  };
  split: {
    initial_train_fraction: number;
    update_fraction: number;
    test_fraction: number;
  };
  training: {
    random_seed: number;
    clients: number;
    initial_rounds: number;
    update_rounds: number;
    local_epochs: number;
    partition: "iid" | "class_skewed";
    participation_fraction: number;
    min_participating_clients: number;
  };
}

export interface ExperimentResults {
  config?: ExperimentConfig;
  training_history?: TrainingHistoryEntry[];
  metrics?: Partial<Record<ModelId, Record<string, DayMetrics>>>;
  data_report?: DataReport;
  model_artifacts?: Partial<Record<ModelId, string>>;
  evaluation?: {
    positive_class?: string;
    false_positive_rate?: string;
    test_sets?: string[];
    feature_count?: number;
    initial_training_rows?: number;
    update_training_rows?: number;
  };
}

export interface RunStatus {
  status: "idle" | "queued" | "running" | "completed" | "failed" | "unavailable";
  run_id?: string | null;
  stage?: string;
  message?: string;
  error?: string;
  model?: ModelId;
  round?: number;
  clients?: Record<string, number>;
  expected_clients?: number;
  selected_clients?: string[];
  successful_participants?: string[];
  failed_participants?: string[];
  aggregation_completed?: boolean;
  client_details?: ClientRoundDetail[];
}

export interface RunResponse {
  id?: string;
  run_id?: string;
  status: "queued" | "running";
  message?: string;
}

export interface MetricsResponse extends DayMetrics {
  model: ModelId;
  test_set: string;
}
