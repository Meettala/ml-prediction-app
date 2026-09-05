"""Versioned trusted model-artifact helpers.

Joblib/pickle artifacts can execute code while loading. Only load artifacts
created by this repository's trusted training pipeline; never accept an
artifact path or bytes from an API user.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import sklearn

from .data import FEATURE_DESCRIPTIONS

ARTIFACT_VERSION = 2
FEATURE_NAMES = tuple(FEATURE_DESCRIPTIONS)
SUPPORTED_MODEL_NAMES = {"linear_baseline", "random_forest"}


class InvalidModelArtifact(RuntimeError):
    """Raised when a trusted local artifact is missing or structurally invalid."""


def build_model_bundle(
    model: Any,
    model_name: str = "random_forest",
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Create the versioned payload saved by the training pipeline."""
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise InvalidModelArtifact("Unsupported model type")
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelArtifact("Model must provide a predict method")
    if scaler is not None and not callable(getattr(scaler, "transform", None)):
        raise InvalidModelArtifact("Scaler must provide a transform method")
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "feature_names": list(FEATURE_NAMES),
        "sklearn_version": sklearn.__version__,
        "model": model,
        "scaler": scaler,
    }


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local model bundle.

    Exact scikit-learn matching is intentional for this demonstration because
    scikit-learn persistence is not a cross-version compatibility contract.
    This check still does not make untrusted pickle/joblib files safe to open.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise InvalidModelArtifact("Model artifact is unavailable")

    try:
        value = joblib.load(artifact_path)
    except Exception as exc:
        raise InvalidModelArtifact("Model artifact could not be loaded") from exc

    if not isinstance(value, dict):
        raise InvalidModelArtifact("Model artifact must be a versioned bundle")
    if value.get("artifact_version") != ARTIFACT_VERSION:
        raise InvalidModelArtifact("Unsupported model artifact version")
    if value.get("model_name") not in SUPPORTED_MODEL_NAMES:
        raise InvalidModelArtifact("Unexpected model type")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise InvalidModelArtifact("Model feature schema does not match the application")

    built_with = value.get("sklearn_version")
    if not isinstance(built_with, str):
        raise InvalidModelArtifact("Model artifact is missing library metadata")
    if built_with != sklearn.__version__:
        raise InvalidModelArtifact(
            "Model was built with a different scikit-learn version; "
            "retrain it with the current environment"
        )

    model = value.get("model")
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelArtifact("Model artifact does not contain a predictor")
    scaler = value.get("scaler")
    if scaler is not None and not callable(getattr(scaler, "transform", None)):
        raise InvalidModelArtifact("Model artifact contains an invalid scaler")
    return value


def predict_from_bundle(bundle: dict[str, Any], features) -> Any:
    """Predict with a validated bundle, applying its training-time scaler if present."""
    model = bundle.get("model")
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelArtifact("Model artifact does not contain a predictor")
    scaler = bundle.get("scaler")
    transformed = scaler.transform(features) if scaler is not None else features
    return model.predict(transformed)


def validate_prediction(value: Any) -> float:
    """Return a finite, non-negative prediction in dataset target units."""
    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidModelArtifact("Model returned a non-numeric prediction") from exc
    if not math.isfinite(prediction) or prediction < 0:
        raise InvalidModelArtifact("Model returned an invalid prediction")
    return prediction
