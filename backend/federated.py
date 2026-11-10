"""Single-process Flower simulation using three simulated study clients."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from backend.centralised import create_model

RoundLogger = Callable[[dict[str, Any]], None]


def partition_indices(
    labels: np.ndarray,
    num_clients: int,
    partition: str,
    seed: int,
) -> list[np.ndarray]:
    """Partition rows deterministically using IID or label-skewed allocation."""
    if num_clients < 2:
        raise ValueError("Federated training requires at least two clients")
    if len(labels) < num_clients:
        raise ValueError("Training samples must be at least the number of clients")
    partition_name = str(partition).strip().lower()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    if partition_name == "iid":
        rng.shuffle(indices)
        shards = np.array_split(indices, num_clients)
    elif partition_name == "class_skewed":
        ordered = indices[np.argsort(labels, kind="stable")]
        shards = np.array_split(ordered, num_clients)
    else:
        raise ValueError("training.partition must be 'iid' or 'class_skewed'")
    if any(len(shard) == 0 for shard in shards):
        raise ValueError("Every federated client must receive at least one sample")
    return [np.asarray(shard, dtype=np.int64) for shard in shards]


class _TrafficClient(fl.client.NumPyClient):
    """Flower client holding only its assigned partition of the training data."""

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model_config: dict[str, Any],
        random_seed: int,
        local_epochs: int,
        client_id: str,
        partition_id: str,
    ) -> None:
        self.x_values = x_values
        self.y_values = y_values
        self.model_config = model_config
        self.random_seed = random_seed
        self.local_epochs = local_epochs
        self.client_id = client_id
        self.partition_id = partition_id
        self.model: SGDClassifier | None = None

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        if self.model is None:
            self.model = create_model(
                self.x_values.shape[1], self.model_config, self.random_seed
            )
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        started = perf_counter()
        self.model = create_model(
            self.x_values.shape[1], self.model_config, self.random_seed
        )
        self.model.coef_ = np.asarray(parameters[0], dtype=np.float64).copy()
        self.model.intercept_ = np.asarray(parameters[1], dtype=np.float64).copy()
        for _ in range(self.local_epochs):
            self.model.partial_fit(
                self.x_values,
                self.y_values,
                classes=np.asarray([0, 1], dtype=np.int8),
            )
        probabilities = self.model.predict_proba(self.x_values)
        return (
            [self.model.coef_.copy(), self.model.intercept_.copy()],
            len(self.y_values),
            {
                "client_id": self.client_id,
                "partition_id": self.partition_id,
                "client_samples": int(len(self.y_values)),
                "local_training_loss": float(
                    log_loss(self.y_values, probabilities, labels=[0, 1])
                ),
                "training_duration_seconds": float(perf_counter() - started),
            },
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        started = perf_counter()
        self.model = create_model(
            self.x_values.shape[1], self.model_config, self.random_seed
        )
        self.model.coef_ = np.asarray(parameters[0], dtype=np.float64).copy()
        self.model.intercept_ = np.asarray(parameters[1], dtype=np.float64).copy()
        predictions = self.model.predict(self.x_values)
        tp = int(np.count_nonzero((self.y_values == 1) & (predictions == 1)))
        fp = int(np.count_nonzero((self.y_values == 0) & (predictions == 1)))
        fn = int(np.count_nonzero((self.y_values == 1) & (predictions == 0)))
        tn = int(np.count_nonzero((self.y_values == 0) & (predictions == 0)))
        accuracy = float(np.mean(predictions == self.y_values))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return 1.0 - accuracy, len(self.y_values), {
            "client_id": self.client_id,
            "partition_id": self.partition_id,
            "client_samples": int(len(self.y_values)),
            "evaluation_duration_seconds": float(perf_counter() - started),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
        }


class _LoggingFedAvg(FedAvg):
    """FedAvg strategy that records per-round evaluation and client samples."""

    def __init__(
        self,
        *args: Any,
        on_round: RoundLogger,
        model_name: str,
        client_ids: list[str],
        partition_ids: list[str],
        partition_sizes: dict[str, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.on_round = on_round
        self.model_name = model_name
        self.client_contributions: dict[str, int] = {}
        self.final_parameters: Parameters | None = None
        self.client_ids = client_ids
        self.partition_ids = partition_ids
        self.partition_sizes = partition_sizes
        self.fit_details: dict[str, dict[str, Any]] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        self.fit_details = {}
        for client, fit_result in results:
            metrics = dict(fit_result.metrics)
            client_id = str(metrics.get("client_id", f"client-{int(client.cid) + 1}"))
            self.fit_details[client_id] = {
                "client_id": client_id,
                "partition_id": str(
                    metrics.get("partition_id", self.partition_ids[int(client.cid)])
                ),
                "samples": int(fit_result.num_examples),
                "local_training_loss": float(
                    metrics.get("local_training_loss", 0.0)
                ),
                "training_duration_seconds": float(
                    metrics.get("training_duration_seconds", 0.0)
                ),
            }
        self.client_contributions = {
            str(fit_result.metrics.get("client_id", f"client-{int(client.cid) + 1}")):
            int(fit_result.num_examples)
            for client, fit_result in results
        }
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.final_parameters = parameters
        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if server_round > 0:
            evaluation_details = {
                str(
                    evaluate_result.metrics.get(
                        "client_id", f"client-{int(client.cid) + 1}"
                    )
                ): {
                    "loss": float(evaluate_result.loss),
                    "evaluation_duration_seconds": float(
                        evaluate_result.metrics.get(
                            "evaluation_duration_seconds", 0.0
                        )
                    ),
                    "accuracy": float(
                        evaluate_result.metrics.get("accuracy", 0.0)
                    ),
                    "precision": float(
                        evaluate_result.metrics.get("precision", 0.0)
                    ),
                    "recall": float(evaluate_result.metrics.get("recall", 0.0)),
                    "f1_score": float(
                        evaluate_result.metrics.get("f1_score", 0.0)
                    ),
                    "false_positive_rate": float(
                        evaluate_result.metrics.get("false_positive_rate", 0.0)
                    ),
                }
                for client, evaluate_result in results
            }
            failed_client_ids = set()
            for failure in failures:
                if (
                    isinstance(failure, tuple)
                    and failure
                    and hasattr(failure[0], "cid")
                ):
                    failed_client_ids.add(f"client-{int(failure[0].cid) + 1}")
            client_details = []
            activity_timestamp = datetime.now(timezone.utc).isoformat()
            for index, client_id in enumerate(self.client_ids):
                fit_detail = self.fit_details.get(client_id, {})
                evaluation_detail = evaluation_details.get(client_id, {})
                failed = client_id in failed_client_ids
                client_details.append(
                    {
                        "client_id": client_id,
                        "partition_id": fit_detail.get(
                            "partition_id", self.partition_ids[index]
                        ),
                        "status": "failed" if failed else (
                            "completed" if evaluation_detail else "training"
                        ),
                        "availability": "unavailable" if failed else (
                            "available" if evaluation_detail else "unknown"
                        ),
                        "selected": client_id in self.fit_details or failed,
                        "samples": int(
                            fit_detail.get(
                                "samples", self.partition_sizes.get(client_id, 0)
                            )
                        ),
                        **fit_detail,
                        **evaluation_detail,
                        "error": (
                            "Flower client failed during this round"
                            if failed
                            else None
                        ),
                        "last_activity": activity_timestamp,
                    }
                )
            self.on_round(
                {
                    "model": self.model_name,
                    "round": int(server_round),
                    "expected_clients": len(self.client_ids),
                    "selected_clients": [
                        client_id
                        for client_id in self.client_ids
                        if client_id in self.fit_details or client_id in failed_client_ids
                    ],
                    "successful_participants": list(evaluation_details),
                    "failed_participants": sorted(failed_client_ids),
                    "aggregation_completed": bool(self.final_parameters is not None),
                    "global_evaluation": {
                        "loss": float(loss or 0.0),
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "f1_score": float(metrics.get("f1_score", 0.0)),
                        "false_positive_rate": float(
                            metrics.get("false_positive_rate", 0.0)
                        ),
                    },
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1_score": float(metrics.get("f1_score", 0.0)),
                    "false_positive_rate": float(
                        metrics.get("false_positive_rate", 0.0)
                    ),
                    "clients": self.client_contributions.copy(),
                    "client_details": client_details,
                }
            )
        return loss, metrics


def train_federated(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    initial_parameters: list[np.ndarray] | None = None,
    model_name: str = "federated",
    on_round: RoundLogger | None = None,
) -> tuple[SGDClassifier, list[dict[str, Any]]]:
    """Run Flower FedAvg over a fixed number of simulated clients and rounds."""
    num_clients = int(training_config["clients"])
    rounds = (
        int(training_config["update_rounds"])
        if model_name.endswith("_updated")
        else int(training_config["initial_rounds"])
    )
    local_epochs = int(training_config["local_epochs"])
    if rounds < 1 or local_epochs < 1:
        raise ValueError("Flower rounds and local epochs must be positive")

    x_values = training_data[feature_columns].to_numpy(dtype=np.float64)
    y_values = training_data[label_column].to_numpy(dtype=np.int8)
    partitions = partition_indices(
        y_values,
        num_clients,
        str(training_config["partition"]),
        int(training_config["random_seed"]),
    )
    client_ids = [f"client-{index + 1}" for index in range(num_clients)]
    partition_ids = [f"partition-{index + 1}" for index in range(num_clients)]
    partition_sizes = {
        client_id: int(len(partition))
        for client_id, partition in zip(client_ids, partitions)
    }
    client_history: list[dict[str, Any]] = []

    def record_round(item: dict[str, Any]) -> None:
        client_history.append(dict(item))
        if on_round is not None:
            on_round(item)

    def client_fn(cid: str) -> fl.client.Client:
        client_index = int(cid)
        client = _TrafficClient(
            x_values[partitions[client_index]],
            y_values[partitions[client_index]],
            model_config,
            int(training_config["random_seed"]) + client_index,
            local_epochs,
            client_ids[client_index],
            partition_ids[client_index],
        )
        return client.to_client()

    initial_model = create_model(
        len(feature_columns),
        model_config,
        int(training_config["random_seed"]),
    )
    parameters = (
        initial_parameters
        if initial_parameters is not None
        else [initial_model.coef_.copy(), initial_model.intercept_.copy()]
    )
    initial_flower_parameters = fl.common.ndarrays_to_parameters(parameters)

    def evaluate_round(
        server_round: int,
        parameters_value: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]]:
        del server_round, config
        model = create_model(
            len(feature_columns),
            model_config,
            int(training_config["random_seed"]),
        )
        model.coef_ = np.asarray(parameters_value[0], dtype=np.float64)
        model.intercept_ = np.asarray(parameters_value[1], dtype=np.float64)
        predictions = model.predict(x_values)
        tp = int(np.count_nonzero((y_values == 1) & (predictions == 1)))
        fp = int(np.count_nonzero((y_values == 0) & (predictions == 1)))
        fn = int(np.count_nonzero((y_values == 1) & (predictions == 0)))
        tn = int(np.count_nonzero((y_values == 0) & (predictions == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        fpr = fp / (fp + tn) if fp + tn else 0.0
        accuracy = (tp + tn) / len(y_values) if len(y_values) else 0.0
        return 1.0 - accuracy, {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
        }

    def aggregate_client_metrics(
        results: list[tuple[int, dict[str, Scalar]]],
    ) -> dict[str, Scalar]:
        total = sum(num_examples for num_examples, _ in results)
        if total == 0:
            return {}
        return {
            metric: float(
                sum(
                    num_examples * float(metrics.get(metric, 0.0))
                    for num_examples, metrics in results
                )
                / total
            )
            for metric in ("precision", "recall", "f1_score", "false_positive_rate")
        }

    strategy = _LoggingFedAvg(
        fraction_fit=float(training_config.get("participation_fraction", 1.0)),
        fraction_evaluate=float(training_config.get("participation_fraction", 1.0)),
        min_fit_clients=int(
            training_config.get("min_participating_clients", num_clients)
        ),
        min_evaluate_clients=int(
            training_config.get("min_participating_clients", num_clients)
        ),
        min_available_clients=num_clients,
        initial_parameters=initial_flower_parameters,
        evaluate_fn=evaluate_round,
        fit_metrics_aggregation_fn=aggregate_client_metrics,
        evaluate_metrics_aggregation_fn=aggregate_client_metrics,
        on_round=record_round,
        model_name=model_name,
        client_ids=client_ids,
        partition_ids=partition_ids,
        partition_sizes=partition_sizes,
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={
            "include_dashboard": False,
            "ignore_reinit_error": True,
            "num_cpus": num_clients,
        },
    )

    if not client_history:
        raise RuntimeError("Flower simulation completed without producing round history")
    final_parameters = strategy.final_parameters
    if final_parameters is None:
        raise RuntimeError("Flower did not return final aggregated model parameters")
    final_arrays = fl.common.parameters_to_ndarrays(final_parameters)
    fitted_model = create_model(
        len(feature_columns), model_config, int(training_config["random_seed"])
    )
    fitted_model.coef_ = final_arrays[0]
    fitted_model.intercept_ = final_arrays[1]
    return fitted_model, client_history
