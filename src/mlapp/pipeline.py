"""End-to-end training, selection, final evaluation and export pipeline."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .artifacts import build_model_bundle
from .data import (
    API_BOUNDS,
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    FINAL_TEST_RANDOM_STATE,
    FINAL_TEST_SIZE,
    UI_BOUNDS,
    VALIDATION_RANDOM_STATE,
    VALIDATION_SIZE_OF_DEVELOPMENT,
    clean_data_with_audit,
    load_data,
    split_train_validation_test,
)
from .train import (
    evaluate_trained,
    feature_importance,
    fit_model,
    predict_trained,
    select_model,
    train_linear_baseline,
    train_random_forest,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EXPORT_PATH = ROOT / "exports" / "metrics.json"
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)
SELECTED_MODEL_FILENAME = "selected_model.joblib"


def _training_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _feature_schema(clean: pd.DataFrame) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name in FEATURE_NAMES:
        api_min, api_max, api_min_exclusive = API_BOUNDS[name]
        ui_min, ui_max = UI_BOUNDS[name]
        schema[name] = {
            "description": FEATURE_DESCRIPTIONS[name],
            "unit": FEATURE_UNITS[name],
            "cleaned_training_range": {
                "min": round(float(clean[name].min()), 6),
                "max": round(float(clean[name].max()), 6),
            },
            "api_demo_range": {
                "min": api_min,
                "max": api_max,
                "min_exclusive": api_min_exclusive,
            },
            "streamlit_convenience_range": {"min": ui_min, "max": ui_max},
        }
    return schema


def run_pipeline(
    models_dir: str | Path = MODELS_DIR,
    export_path: str | Path = EXPORT_PATH,
) -> dict[str, Any]:
    """Train fixed candidates, select on validation, then evaluate once on final test."""
    model_directory = Path(models_dir)
    metrics_path = Path(export_path)

    raw = load_data()
    clean, cleaning_audit = clean_data_with_audit(raw)
    splits = split_train_validation_test(clean)

    linear_validation = train_linear_baseline(
        splits.X_train,
        splits.y_train,
        splits.X_validation,
        splits.y_validation,
    )
    forest_validation = train_random_forest(
        splits.X_train,
        splits.y_train,
        splits.X_validation,
        splits.y_validation,
    )
    validation_metrics = {
        "linear_baseline": linear_validation.metrics,
        "random_forest": forest_validation.metrics,
    }
    selected_name = select_model(validation_metrics)

    development_index = sorted(
        [*splits.X_train.index.tolist(), *splits.X_validation.index.tolist()]
    )
    X_development = clean.loc[development_index, FEATURE_NAMES]
    y_development = clean.loc[development_index, "MedHouseVal"]
    selected_model = fit_model(selected_name, X_development, y_development)
    final_test_metrics = evaluate_trained(selected_model, splits.X_test, splits.y_test)

    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        build_model_bundle(
            selected_model.model,
            model_name=selected_model.name,
            scaler=selected_model.scaler,
        ),
        model_directory / SELECTED_MODEL_FILENAME,
    )

    importances = feature_importance(selected_model, FEATURE_NAMES)
    result: dict[str, Any] = {
        "schema_version": 2,
        "dataset": {
            "name": "California Housing",
            "source": "scikit-learn built-in dataset derived from 1990 US Census block groups",
            "target": "MedHouseVal",
            "target_description": "Median block-group house value in $100,000 dataset units",
        },
        "cleaning": {
            **cleaning_audit.to_dict(),
            "filters": {
                "AveRooms": "remove rows where AveRooms >= 30",
                "AveOccup": "after prior filtering, remove rows where AveOccup >= 15",
            },
            "interpretation": "fixed transparent demonstration filters; not statistically optimal outlier treatment",
        },
        "split": {
            "method": "deterministic train/validation/final-test",
            "final_test_fraction": FINAL_TEST_SIZE,
            "validation_fraction_of_development": VALIDATION_SIZE_OF_DEVELOPMENT,
            "final_test_random_state": FINAL_TEST_RANDOM_STATE,
            "validation_random_state": VALIDATION_RANDOM_STATE,
            "row_counts": splits.row_counts(),
            "development_refit_rows": int(len(X_development)),
            "purpose": {
                "train": "fit fixed candidate models",
                "validation": "compare fixed candidates and select serving model",
                "test": "one final held-out evaluation after model selection",
            },
        },
        "selection": {
            "candidate_models": ["linear_baseline", "random_forest"],
            "criterion": "lowest validation RMSE",
            "tie_breaker": "linear_baseline (simpler fixed candidate)",
            "selected_model": selected_name,
            "final_test_used_for_selection": False,
        },
        "model_parameters": {
            "linear_baseline": linear_validation.parameters,
            "random_forest": forest_validation.parameters,
        },
        "validation_metrics": validation_metrics,
        "final_test_metrics": {"model": selected_name, **final_test_metrics},
        "training_environment": _training_environment(),
        "feature_schema": _feature_schema(clean),
        "feature_importance_selected_model": importances,
        "sample_predictions": _sample_predictions(
            splits.X_test,
            splits.y_test,
            selected_model,
        ),
        "limitations": [
            "The dataset reflects 1990 census conditions, not current housing prices.",
            "Rows are block-group aggregates, not individual properties.",
            "Final metrics are from one deterministic held-out split, not temporal validation.",
            "R² is variance explained on this historical split, not prediction accuracy or confidence.",
            "Random Forest impurity importance is model inspection, not causal explanation.",
            "Illustrative portfolio project only — not a valuation or financial recommendation.",
        ],
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _sample_predictions(X_test, y_test, model, n: int = 5) -> list[dict[str, Any]]:
    sample_size = min(max(int(n), 0), len(X_test))
    if sample_size == 0:
        return []

    sample = X_test.sample(n=sample_size, random_state=1)
    predictions = predict_trained(model, sample)
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
                "predicted_value_100k": round(float(prediction), 3),
                "actual_value_100k": round(float(actual), 3),
            }
        )
    return rows


if __name__ == "__main__":
    analysis = run_pipeline()
    print("Training environment:", analysis["training_environment"])
    print("Cleaning audit:", analysis["cleaning"])
    print("Split:", analysis["split"])
    print("Validation metrics:", analysis["validation_metrics"])
    print("Selected model:", analysis["selection"]["selected_model"])
    print("Final test metrics:", analysis["final_test_metrics"])
    print(f"Exported to {EXPORT_PATH}")
