"""Fixed-candidate regression training and deterministic evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 14,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class TrainedModel:
    name: str
    model: Any
    scaler: StandardScaler | None
    metrics: dict[str, float]
    parameters: dict[str, Any]


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def predict_trained(model: TrainedModel, features):
    transformed = model.scaler.transform(features) if model.scaler is not None else features
    return model.model.predict(transformed)


def evaluate_trained(model: TrainedModel, features, target) -> dict[str, float]:
    return evaluate_predictions(target, predict_trained(model, features))


def fit_linear_baseline(X_train, y_train) -> TrainedModel:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return TrainedModel(
        name="linear_baseline",
        model=model,
        scaler=scaler,
        metrics={},
        parameters={"fit_intercept": True, "standard_scaler": True},
    )


def fit_random_forest(X_train, y_train) -> TrainedModel:
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMETERS)
    model.fit(X_train, y_train)
    return TrainedModel(
        name="random_forest",
        model=model,
        scaler=None,
        metrics={},
        parameters=dict(RANDOM_FOREST_PARAMETERS),
    )


def fit_model(name: str, X_train, y_train) -> TrainedModel:
    if name == "linear_baseline":
        return fit_linear_baseline(X_train, y_train)
    if name == "random_forest":
        return fit_random_forest(X_train, y_train)
    raise ValueError(f"Unsupported model candidate: {name}")


def train_linear_baseline(X_train, y_train, X_eval, y_eval) -> TrainedModel:
    """Fit the baseline on training rows and evaluate on the supplied partition."""
    trained = fit_linear_baseline(X_train, y_train)
    trained.metrics = evaluate_trained(trained, X_eval, y_eval)
    return trained


def train_random_forest(X_train, y_train, X_eval, y_eval) -> TrainedModel:
    """Fit Random Forest on training rows and evaluate on the supplied partition."""
    trained = fit_random_forest(X_train, y_train)
    trained.metrics = evaluate_trained(trained, X_eval, y_eval)
    return trained


def select_model(validation_metrics: dict[str, dict[str, float]]) -> str:
    """Select the fixed candidate with the lowest validation RMSE.

    An exact RMSE tie selects the simpler Linear Regression baseline.
    """
    required = {"linear_baseline", "random_forest"}
    if set(validation_metrics) != required:
        raise ValueError("Validation metrics must contain exactly the fixed candidate models")
    for name in required:
        rmse = validation_metrics[name].get("rmse")
        if not isinstance(rmse, int | float) or not np.isfinite(float(rmse)):
            raise ValueError(f"Validation RMSE for {name} must be finite")

    if validation_metrics["random_forest"]["rmse"] < validation_metrics["linear_baseline"]["rmse"]:
        return "random_forest"
    return "linear_baseline"


def feature_importance(rf_model: TrainedModel, feature_names: list[str]) -> list[dict[str, float | str]]:
    if rf_model.name != "random_forest":
        return []
    importances = rf_model.model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in order
    ]
