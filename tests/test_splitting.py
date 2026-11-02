"""Tests for reproducible and leakage-free random data partitions."""

import pandas as pd
import pytest

from backend.data_prep import fit_feature_scaler, transform_features
from backend.splitting import split_random


@pytest.fixture
def split_config() -> dict:
    return {
        "dataset": {"binary_label_column": "binary_label"},
        "split": {
            "strategy": "stratified_random",
            "initial_train_fraction": 0.7,
            "update_fraction": 0.1,
            "test_fraction": 0.2,
            "test_set_name": "Random holdout",
        },
        "training": {"random_seed": 42},
    }


def _balanced_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_size": range(100),
            "binary_label": [0] * 80 + [1] * 20,
            "attack_label": ["Normal Traffic"] * 80 + ["DoS"] * 20,
        }
    )


def test_random_split_is_disjoint_stratified_and_reproducible(
    split_config: dict,
) -> None:
    data = _balanced_data()

    initial, update, tests = split_random(data, split_config)
    repeat_initial, repeat_update, repeat_tests = split_random(data, split_config)
    test = tests["Random holdout"]

    assert (len(initial), len(update), len(test)) == (70, 10, 20)
    assert initial["binary_label"].mean() == pytest.approx(0.2)
    assert update["binary_label"].mean() == pytest.approx(0.2)
    assert test["binary_label"].mean() == pytest.approx(0.2)
    assert set(initial["flow_size"]).isdisjoint(update["flow_size"])
    assert set(initial["flow_size"]).isdisjoint(test["flow_size"])
    assert set(update["flow_size"]).isdisjoint(test["flow_size"])
    assert initial.equals(repeat_initial)
    assert update.equals(repeat_update)
    assert test.equals(repeat_tests["Random holdout"])


def test_random_split_rejects_fractions_that_do_not_sum_to_one(
    split_config: dict,
) -> None:
    split_config["split"]["test_fraction"] = 0.3
    with pytest.raises(ValueError, match="sum to 1"):
        split_random(_balanced_data(), split_config)


def test_scaler_is_fitted_on_initial_training_data_only() -> None:
    training = pd.DataFrame({"flow_size": [1.0, 3.0], "binary_label": [0, 1]})
    later_test_data = pd.DataFrame({"flow_size": [1000], "binary_label": [1]})

    scaler, feature_columns = fit_feature_scaler(training, ["binary_label"])
    transformed_test = transform_features(later_test_data, scaler, feature_columns)

    assert scaler.mean_.tolist() == [2.0]
    assert transformed_test.loc[0, "flow_size"] > 500
