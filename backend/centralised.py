"""Centralised logistic-regression training for the study."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


def create_model(
    feature_count: int,
    model_config: dict[str, Any],
    random_seed: int,
) -> SGDClassifier:
    """Create an averageable logistic model with deterministic initialization."""
    if model_config.get("type") != "logistic_regression":
        raise ValueError(
            "This implementation supports model.type='logistic_regression' only"
        )
    parameters = model_config.get("hyperparameters", {})
    penalty = str(parameters.get("penalty", "l2"))
    if penalty not in {"l2", "l1", "elasticnet"}:
        raise ValueError(f"Unsupported logistic-regression penalty: {penalty}")
    c_value = float(parameters.get("C", 1.0))
    eta0 = float(parameters.get("eta0", 0.01))
    if c_value <= 0 or not np.isfinite(c_value):
        raise ValueError("Model regularisation strength C must be positive")
    if eta0 <= 0 or not np.isfinite(eta0):
        raise ValueError("Model learning rate eta0 must be positive")
    estimator = SGDClassifier(
        loss="log_loss",
        penalty=penalty,
        alpha=1.0 / c_value,
        learning_rate=str(parameters.get("learning_rate", "constant")),
        eta0=eta0,
        tol=None,
        random_state=random_seed,
        average=False,
    )
    estimator.partial_fit(
        np.zeros((1, feature_count), dtype=np.float64),
        np.asarray([0], dtype=np.int8),
        classes=np.asarray([0, 1], dtype=np.int8),
    )
    estimator.coef_.fill(0.0)
    estimator.intercept_.fill(0.0)
    return estimator


def fit_epochs(
    model: SGDClassifier,
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    epochs: int,
) -> SGDClassifier:
    """Continue fitting one model for a fixed number of full data passes."""
    x_values = np.asarray(features, dtype=np.float64)
    y_values = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(x_values) != len(y_values) or not len(y_values):
        raise ValueError("Training data must contain matching, non-empty features and labels")
    if epochs < 1:
        raise ValueError("The number of training epochs must be positive")
    for _ in range(epochs):
        model.partial_fit(x_values, y_values, classes=np.asarray([0, 1], dtype=np.int8))
    return model
