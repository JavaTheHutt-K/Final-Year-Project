"""Run the configured centralised and federated intrusion detection study."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from backend.centralised import create_model, fit_epochs
from backend.data_prep import (
    drop_training_constant_columns,
    fit_feature_scaler,
    load_config,
    prepare_data,
    save_data_report,
    transform_features,
)
from backend.evaluate import evaluate_predictions
from backend.federated import train_federated
from backend.splitting import split_random

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, dict[str, Any]], None]


def _set_random_seeds(seed: int) -> None:
    """Set Python and NumPy random seeds used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)


def _feature_frame(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return a numeric model input frame without study labels or day metadata."""
    return data.loc[:, feature_columns]


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Send experiment progress to a caller such as the FastAPI status store."""
    if callback is not None:
        callback(stage, detail or {})


def run_experiment(
    config_path: str | Path = "config.yaml",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run preparation, training, model updating, and holdout evaluation."""
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    seed = int(config["training"]["random_seed"])
    _set_random_seeds(seed)

    _notify(progress_callback, "preparing_data")
    data, data_report = prepare_data(config_file)
    initial_data, update_data, test_data = split_random(data, config)

    dataset_config = config["dataset"]
    label_column = dataset_config["binary_label_column"]
    excluded_columns = [
        dataset_config["label_column"],
        dataset_config["attack_label_column"],
        dataset_config["binary_label_column"],
    ]
    preprocessing_config = config.get("preprocessing", {})
    if bool(preprocessing_config.get("drop_constant_columns", False)):
        data, constant_columns = drop_training_constant_columns(
            data,
            initial_data,
            excluded_columns,
        )
    else:
        constant_columns = []
    initial_data = initial_data.drop(columns=constant_columns).reset_index(drop=True)
    update_data = update_data.drop(columns=constant_columns).reset_index(drop=True)
    test_data = {
        test_set: test_frame.drop(columns=constant_columns).reset_index(drop=True)
        for test_set, test_frame in test_data.items()
    }
    data_report["constant_columns_dropped"] = constant_columns
    data_report["totals"]["rows_after_cleaning"] = int(len(data))
    data_report["totals"]["columns_after_cleaning"] = int(len(data.columns))
    data_report["totals"]["feature_columns_after_cleaning"] = len(
        [
            column
            for column in data.select_dtypes(include=[np.number]).columns
            if column not in excluded_columns
        ]
    )
    data_report["totals"]["constant_columns_dropped"] = len(constant_columns)
    data_report["partitions"] = {
        "initial_training": {
            "rows": int(len(initial_data)),
            "class_balance": {
                "benign": int((initial_data[label_column] == 0).sum()),
                "malicious": int((initial_data[label_column] == 1).sum()),
            },
        },
        "update": {
            "rows": int(len(update_data)),
            "class_balance": {
                "benign": int((update_data[label_column] == 0).sum()),
                "malicious": int((update_data[label_column] == 1).sum()),
            },
        },
        "test": {
            "rows": int(sum(len(frame) for frame in test_data.values())),
            "class_balance": {
                "benign": int(
                    sum((frame[label_column] == 0).sum() for frame in test_data.values())
                ),
                "malicious": int(
                    sum((frame[label_column] == 1).sum() for frame in test_data.values())
                ),
            },
        },
    }
    save_data_report(data_report, config_file)
    scaler, feature_columns = fit_feature_scaler(
        initial_data,
        excluded_columns,
        str(preprocessing_config.get("scaler", "standard")),
    )
    initial_data = transform_features(initial_data, scaler, feature_columns)
    update_data = transform_features(update_data, scaler, feature_columns)
    test_data = {
        day: transform_features(day_data, scaler, feature_columns)
        for day, day_data in test_data.items()
    }

    training_config = config["training"]
    initial_epochs = (
        int(training_config["initial_rounds"])
        * int(training_config["local_epochs"])
    )
    update_epochs = (
        int(training_config["update_rounds"])
        * int(training_config["local_epochs"])
    )
    history: list[dict[str, Any]] = []

    def add_federated_round(item: dict[str, Any]) -> None:
        history.append(dict(item))
        _notify(
            progress_callback,
            "federated_training",
            item,
        )

    _notify(progress_callback, "centralised_training")
    central_model = create_model(
        len(feature_columns), config["model"], seed
    )
    fit_epochs(
        central_model,
        _feature_frame(initial_data, feature_columns),
        initial_data[label_column],
        initial_epochs,
    )

    _notify(progress_callback, "federated_training")
    federated_model, _ = train_federated(
        initial_data,
        feature_columns,
        label_column,
        config["model"],
        training_config,
        model_name="federated",
        on_round=add_federated_round,
    )

    central_updated_model = deepcopy(central_model)
    combined_update = pd.concat(
        [initial_data, update_data],
        ignore_index=True,
        sort=False,
    )
    _notify(progress_callback, "updating_centralised_model")
    fit_epochs(
        central_updated_model,
        _feature_frame(combined_update, feature_columns),
        combined_update[label_column],
        update_epochs,
    )

    _notify(progress_callback, "updating_federated_model")
    federated_updated_model, _ = train_federated(
        update_data,
        feature_columns,
        label_column,
        config["model"],
        training_config,
        initial_parameters=[
            federated_model.coef_.copy(),
            federated_model.intercept_.copy(),
        ],
        model_name="federated_updated",
        on_round=add_federated_round,
    )

    models: dict[str, SGDClassifier] = {
        "federated": federated_model,
        "centralised": central_model,
        "federated_updated": federated_updated_model,
        "centralised_updated": central_updated_model,
    }
    test_sets = list(test_data)
    metrics_by_model: dict[str, dict[str, Any]] = {
        model_name: {} for model_name in models
    }

    _notify(progress_callback, "evaluating")
    for test_set in test_sets:
        test_frame = test_data[test_set]
        x_test = _feature_frame(test_frame, feature_columns)
        y_test = test_frame[label_column].to_numpy(dtype=np.int8)
        attack_labels = test_frame[dataset_config["attack_label_column"]].to_numpy(
            dtype=str
        )
        for model_name, model in models.items():
            predictions = model.predict(x_test)
            metrics_by_model[model_name][test_set] = evaluate_predictions(
                y_test,
                predictions,
                attack_labels,
                str(dataset_config["benign_label"]),
            )

    root = config_file.parent
    results_dir = Path(config["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = results_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "binary_label_column": label_column,
            },
            artifact_dir / f"{model_name}.joblib",
        )

    results = {
        "config": config,
        "training_history": history,
        "metrics": metrics_by_model,
        "data_report": data_report,
        "model_artifacts": {
            model_name: str(artifact_dir / f"{model_name}.joblib")
            for model_name in models
        },
        "evaluation": {
            "positive_class": "malicious",
            "false_positive_rate": "FP / (FP + TN)",
            "test_sets": test_sets,
            "feature_count": len(feature_columns),
            "initial_training_rows": int(len(initial_data)),
            "update_training_rows": int(len(update_data)),
            "test_rows": int(sum(len(frame) for frame in test_data.values())),
        },
    }
    results_path = Path(config["paths"]["results_file"])
    if not results_path.is_absolute():
        results_path = root / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    _notify(progress_callback, "completed", {"results_file": str(results_path)})
    logger.info("Experiment results written to %s", results_path)
    return results


def main() -> None:
    """Run the experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the root YAML configuration (default: config.yaml)",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_experiment(arguments.config)


if __name__ == "__main__":
    main()
