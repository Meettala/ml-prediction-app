"""Versioned model artifact helpers.

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

ARTIFACT_VERSION = 1
MODEL_NAME = "random_forest"
FEATURE_NAMES = tuple(FEATURE_DESCRIPTIONS)


class InvalidModelArtifact(RuntimeError):
    """Raised when a trusted local artifact is missing or structurally invalid."""


def build_model_bundle(model: Any) -> dict[str, Any]:
    """Create the versioned payload saved by the training pipeline."""
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelArtifact("Model must provide a predict method")
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": MODEL_NAME,
        "feature_names": list(FEATURE_NAMES),
        "sklearn_version": sklearn.__version__,
        "model": model,
    }


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local model bundle.

    This validation detects missing, corrupt, stale or incompatible metadata;
    it does not make untrusted pickle/joblib files safe to open.
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
    if value.get("model_name") != MODEL_NAME:
        raise InvalidModelArtifact("Unexpected model type")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise InvalidModelArtifact("Model feature schema does not match the application")
    if not isinstance(value.get("sklearn_version"), str):
        raise InvalidModelArtifact("Model artifact is missing library metadata")

    model = value.get("model")
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelArtifact("Model artifact does not contain a predictor")
    return value


def validate_prediction(value: Any) -> float:
    """Return a finite, non-negative prediction in dataset target units."""
    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidModelArtifact("Model returned a non-numeric prediction") from exc
    if not math.isfinite(prediction) or prediction < 0:
        raise InvalidModelArtifact("Model returned an invalid prediction")
    return prediction
