"""
Training: a baseline Linear Regression (interpretable, used for the
client-side interactive demo on the portfolio site) and a stronger Random
Forest (used for the "real" reported accuracy and the FastAPI/Streamlit
demos). Reporting both, honestly, rather than only showing the best number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    name: str
    model: object
    scaler: StandardScaler | None
    metrics: dict


def _evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_linear_baseline(X_train, y_train, X_test, y_test) -> TrainedModel:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    return TrainedModel(
        name="linear_baseline",
        model=model,
        scaler=scaler,
        metrics=_evaluate(y_test, preds),
    )


def train_random_forest(X_train, y_train, X_test, y_test) -> TrainedModel:
    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return TrainedModel(
        name="random_forest",
        model=model,
        scaler=None,
        metrics=_evaluate(y_test, preds),
    )


def feature_importance(rf_model: TrainedModel, feature_names: list[str]) -> list[dict]:
    importances = rf_model.model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in order
    ]
