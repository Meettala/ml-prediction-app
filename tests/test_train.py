from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mlapp.data import FEATURE_DESCRIPTIONS
from src.mlapp.train import (
    RANDOM_FOREST_PARAMETERS,
    select_model,
    train_linear_baseline,
    train_random_forest,
)


def synthetic_regression_data(rows: int = 80):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            name: rng.normal(loc=index + 2.0, scale=0.5, size=rows)
            for index, name in enumerate(FEATURE_DESCRIPTIONS)
        }
    )
    y = 1.5 * X["MedInc"] - 0.4 * X["AveOccup"] + rng.normal(0, 0.05, rows)
    return X.iloc[:60], X.iloc[60:], y.iloc[:60], y.iloc[60:]


def test_fixed_candidates_train_and_report_finite_validation_metrics():
    X_train, X_validation, y_train, y_validation = synthetic_regression_data()

    linear = train_linear_baseline(X_train, y_train, X_validation, y_validation)
    forest = train_random_forest(X_train, y_train, X_validation, y_validation)

    for trained in (linear, forest):
        assert set(trained.metrics) == {"rmse", "mae", "r2"}
        assert all(np.isfinite(value) for value in trained.metrics.values())

    assert forest.parameters == RANDOM_FOREST_PARAMETERS


def test_selection_uses_validation_rmse_and_linear_tie_breaker():
    assert (
        select_model(
            {
                "linear_baseline": {"rmse": 0.50},
                "random_forest": {"rmse": 0.40},
            }
        )
        == "random_forest"
    )
    assert (
        select_model(
            {
                "linear_baseline": {"rmse": 0.40},
                "random_forest": {"rmse": 0.40},
            }
        )
        == "linear_baseline"
    )


def test_selection_rejects_missing_or_non_finite_validation_evidence():
    with pytest.raises(ValueError, match="exactly"):
        select_model({"random_forest": {"rmse": 0.4}})
    with pytest.raises(ValueError, match="finite"):
        select_model(
            {
                "linear_baseline": {"rmse": 0.5},
                "random_forest": {"rmse": np.nan},
            }
        )
