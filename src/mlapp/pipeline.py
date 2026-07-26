"""End-to-end training, evaluation and export pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .artifacts import build_model_bundle
from .data import FEATURE_DESCRIPTIONS, clean_data, load_data, split_data
from .train import feature_importance, train_linear_baseline, train_random_forest

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EXPORT_PATH = ROOT / "exports" / "metrics.json"
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)


def run_pipeline(
    models_dir: str | Path = MODELS_DIR,
    export_path: str | Path = EXPORT_PATH,
) -> dict[str, Any]:
    """Train deterministic models and write validated local artifacts."""
    model_directory = Path(models_dir)
    metrics_path = Path(export_path)

    raw = load_data()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)

    linear = train_linear_baseline(X_train, y_train, X_test, y_test)
    forest = train_random_forest(X_train, y_train, X_test, y_test)

    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        build_model_bundle(forest.model),
        model_directory / "random_forest.joblib",
    )
    joblib.dump(
        {
            "artifact_version": 1,
            "model_name": "linear_baseline",
            "feature_names": FEATURE_NAMES,
            "model": linear.model,
            "scaler": linear.scaler,
        },
        model_directory / "linear_baseline.joblib",
    )

    importances = feature_importance(forest, FEATURE_NAMES)
    coef_std = linear.model.coef_
    intercept_std = linear.model.intercept_
    means = linear.scaler.mean_
    scales = linear.scaler.scale_
    raw_coefs = coef_std / scales
    raw_intercept = intercept_std - sum(
        coef_std[index] * means[index] / scales[index]
        for index in range(len(coef_std))
    )

    result: dict[str, Any] = {
        "dataset": "California Housing (1990 US Census block groups)",
        "rows_after_cleaning": int(len(clean)),
        "target_description": "Median house value per block group, in $100,000s",
        "features": FEATURE_DESCRIPTIONS,
        "models": {
            "linear_baseline": linear.metrics,
            "random_forest": forest.metrics,
        },
        "feature_importance_random_forest": importances,
        "linear_model_raw_coefficients": {
            FEATURE_NAMES[index]: round(float(raw_coefs[index]), 6)
            for index in range(len(FEATURE_NAMES))
        },
        "linear_model_intercept": round(float(raw_intercept), 6),
        "sample_predictions": _sample_predictions(X_test, y_test, forest),
        "limitations": [
            "Trained on 1990 census data — not current housing prices.",
            "Block-group level, not individual-property level.",
            "Random Forest metrics are on a held-out test set, not live data.",
            "Illustrative portfolio project only — not financial advice.",
        ],
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _sample_predictions(X_test, y_test, forest, n: int = 5) -> list[dict[str, Any]]:
    sample_size = min(max(int(n), 0), len(X_test))
    if sample_size == 0:
        return []

    sample = X_test.sample(n=sample_size, random_state=1)
    predictions = forest.model.predict(sample)
    rows: list[dict[str, Any]] = []
    for (_, row), prediction, actual in zip(
        sample.iterrows(),
        predictions,
        y_test.loc[sample.index],
        strict=True,
    ):
        rows.append(
            {
                "features": {key: round(float(value), 2) for key, value in row.items()},
                "predicted_value_100k": round(float(prediction), 2),
                "actual_value_100k": round(float(actual), 2),
            }
        )
    return rows


if __name__ == "__main__":
    analysis = run_pipeline()
    print("Random Forest:", analysis["models"]["random_forest"])
    print("Linear baseline:", analysis["models"]["linear_baseline"])
    print(f"Exported to {EXPORT_PATH}")
