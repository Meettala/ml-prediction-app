"""
End-to-end pipeline: load -> clean -> split -> train (linear + random
forest) -> evaluate -> save model artifacts -> export a JSON summary
consumed by the Streamlit app, the FastAPI service, and the static
portfolio-site demo.

Usage:
    python -m src.mlapp.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .data import FEATURE_DESCRIPTIONS, clean_data, load_data, split_data
from .train import feature_importance, train_linear_baseline, train_random_forest

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EXPORT_PATH = ROOT / "exports" / "metrics.json"

FEATURE_NAMES = list(FEATURE_DESCRIPTIONS.keys())


def run_pipeline() -> dict:
    raw = load_data()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)

    linear = train_linear_baseline(X_train, y_train, X_test, y_test)
    forest = train_random_forest(X_train, y_train, X_test, y_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(forest.model, MODELS_DIR / "random_forest.joblib")
    joblib.dump({"model": linear.model, "scaler": linear.scaler}, MODELS_DIR / "linear_baseline.joblib")

    importances = feature_importance(forest, FEATURE_NAMES)

    # Coefficients for the client-side JS reproduction of the linear model
    # (used on the portfolio site's interactive demo). Because the linear
    # model was trained on standardized features, we fold the scaler's
    # mean/scale into the coefficients so the JS demo can take raw feature
    # values directly.
    coef_std = linear.model.coef_
    intercept_std = linear.model.intercept_
    means = linear.scaler.mean_
    scales = linear.scaler.scale_
    raw_coefs = coef_std / scales
    raw_intercept = intercept_std - sum(coef_std[i] * means[i] / scales[i] for i in range(len(coef_std)))

    result = {
        "dataset": "California Housing (scikit-learn built-in, 1990 US Census block groups)",
        "rows_after_cleaning": int(len(clean)),
        "target_description": "Median house value per block group, in $100,000s",
        "features": FEATURE_DESCRIPTIONS,
        "models": {
            "linear_baseline": linear.metrics,
            "random_forest": forest.metrics,
        },
        "feature_importance_random_forest": importances,
        "linear_model_raw_coefficients": {
            FEATURE_NAMES[i]: round(float(raw_coefs[i]), 6) for i in range(len(FEATURE_NAMES))
        },
        "linear_model_intercept": round(float(raw_intercept), 6),
        "sample_predictions": _sample_predictions(X_test, y_test, forest),
        "limitations": [
            "Trained on 1990 census data — not current housing prices.",
            "Block-group level, not individual-property level.",
            "Random Forest metrics are on a held-out test set, not live data.",
            "Illustrative portfolio project only — not financial or investment advice.",
        ],
    }

    EXPORT_PATH.parent.mkdir(exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(result, indent=2))
    return result


def _sample_predictions(X_test, y_test, forest, n: int = 5) -> list[dict]:
    sample = X_test.sample(n=n, random_state=1)
    preds = forest.model.predict(sample)
    rows = []
    for (idx, row), pred, actual in zip(sample.iterrows(), preds, y_test.loc[sample.index]):
        rows.append(
            {
                "features": {k: round(float(v), 2) for k, v in row.items()},
                "predicted_value_100k": round(float(pred), 2),
                "actual_value_100k": round(float(actual), 2),
            }
        )
    return rows


if __name__ == "__main__":
    result = run_pipeline()
    print("Random Forest:", result["models"]["random_forest"])
    print("Linear baseline:", result["models"]["linear_baseline"])
    print(f"Exported to {EXPORT_PATH}")
