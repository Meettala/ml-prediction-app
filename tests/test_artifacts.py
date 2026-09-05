from __future__ import annotations

import math

import joblib
import numpy as np
import pytest

from src.mlapp.artifacts import (
    FEATURE_NAMES,
    InvalidModelArtifact,
    build_model_bundle,
    load_model_bundle,
    predict_from_bundle,
    validate_prediction,
)


class DummyModel:
    def predict(self, frame):
        return [2.5] * len(frame)


class DoubleScaler:
    def transform(self, frame):
        return np.asarray(frame) * 2


class RecordingModel:
    def __init__(self):
        self.last = None

    def predict(self, frame):
        self.last = frame
        return [1.0] * len(frame)


def test_round_trips_valid_model_bundle(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(build_model_bundle(DummyModel()), path)

    bundle = load_model_bundle(path)

    assert bundle["feature_names"] == list(FEATURE_NAMES)
    assert bundle["model_name"] == "random_forest"
    assert bundle["model"].predict([{}]) == [2.5]


def test_bundle_prediction_applies_training_scaler_when_present():
    model = RecordingModel()
    bundle = build_model_bundle(model, model_name="linear_baseline", scaler=DoubleScaler())

    predictions = predict_from_bundle(bundle, [[1.0, 2.0]])

    assert predictions == [1.0]
    assert model.last.tolist() == [[2.0, 4.0]]


def test_rejects_missing_and_unversioned_artifacts(tmp_path):
    with pytest.raises(InvalidModelArtifact, match="unavailable"):
        load_model_bundle(tmp_path / "missing.joblib")

    path = tmp_path / "legacy.joblib"
    joblib.dump(DummyModel(), path)
    with pytest.raises(InvalidModelArtifact, match="versioned bundle"):
        load_model_bundle(path)


def test_rejects_feature_schema_mismatch(tmp_path):
    path = tmp_path / "wrong-schema.joblib"
    bundle = build_model_bundle(DummyModel())
    bundle["feature_names"] = ["unexpected"]
    joblib.dump(bundle, path)

    with pytest.raises(InvalidModelArtifact, match="feature schema"):
        load_model_bundle(path)


def test_rejects_different_scikit_learn_version(tmp_path):
    path = tmp_path / "wrong-sklearn.joblib"
    bundle = build_model_bundle(DummyModel())
    bundle["sklearn_version"] = "0.0.0"
    joblib.dump(bundle, path)

    with pytest.raises(InvalidModelArtifact, match="different scikit-learn version"):
        load_model_bundle(path)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1, "not-a-number"])
def test_rejects_invalid_predictions(value):
    with pytest.raises(InvalidModelArtifact):
        validate_prediction(value)


def test_accepts_finite_non_negative_prediction():
    assert validate_prediction("2.75") == 2.75
