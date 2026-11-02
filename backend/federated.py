"""Single-process Flower simulation using three simulated study clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from sklearn.linear_model import SGDClassifier

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
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    if partition.lower() == "iid":
        rng.shuffle(indices)
        shards = np.array_split(indices, num_clients)
    elif partition.lower() in {"class_skewed", "class-skewed"}:
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
    ) -> None:
        self.x_values = x_values
        self.y_values = y_values
        self.model_config = model_config
        self.random_seed = random_seed
        self.local_epochs = local_epochs
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
        return (
            [self.model.coef_.copy(), self.model.intercept_.copy()],
            len(self.y_values),
            {"client_samples": int(len(self.y_values))},
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.on_round = on_round
        self.model_name = model_name
        self.client_contributions: dict[str, int] = {}
        self.final_parameters: Parameters | None = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        self.client_contributions = {
            f"Client {int(client.cid) + 1}": int(fit_result.num_examples)
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
            self.on_round(
                {
                    "model": self.model_name,
                    "round": int(server_round),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1_score": float(metrics.get("f1_score", 0.0)),
                    "false_positive_rate": float(
                        metrics.get("false_positive_rate", 0.0)
                    ),
                    "clients": self.client_contributions.copy(),
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
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_flower_parameters,
        evaluate_fn=evaluate_round,
        fit_metrics_aggregation_fn=aggregate_client_metrics,
        evaluate_metrics_aggregation_fn=aggregate_client_metrics,
        on_round=record_round,
        model_name=model_name,
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
