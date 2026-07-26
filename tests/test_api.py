from __future__ import annotations

import math

from fastapi.testclient import TestClient

import api.main as api_main
from src.mlapp.artifacts import FEATURE_NAMES


class RecordingModel:
    def __init__(self, prediction=2.5):
        self.prediction = prediction
        self.columns = None

    def predict(self, frame):
        self.columns = list(frame.columns)
        return [self.prediction]


VALID_PAYLOAD = {
    "MedInc": 5.0,
    "HouseAge": 20,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0,
}


def setup_function():
    api_main._model_bundle = None


def test_health_reports_artifact_presence():
    response = TestClient(api_main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_artifact_present"], bool)


def test_predict_uses_canonical_feature_order():
    model = RecordingModel()
    api_main._model_bundle = {"model": model, "model_name": "random_forest"}

    response = TestClient(api_main.app).post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert model.columns == list(FEATURE_NAMES)
    assert response.json()["predicted_value_usd"] == 250000.0


def test_rejects_extra_non_finite_and_out_of_range_inputs():
    client = TestClient(api_main.app)

    assert client.post(
        "/predict",
        json={**VALID_PAYLOAD, "unexpected": 1},
    ).status_code == 422
    assert client.post(
        "/predict",
        json={**VALID_PAYLOAD, "MedInc": "NaN"},
    ).status_code == 422
    assert client.post(
        "/predict",
        json={**VALID_PAYLOAD, "AveRooms": 100},
    ).status_code == 422


def test_returns_safe_error_for_invalid_model_output():
    api_main._model_bundle = {
        "model": RecordingModel(math.nan),
        "model_name": "random_forest",
    }

    response = TestClient(api_main.app).post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["detail"] == "The prediction could not be produced safely."


def test_returns_503_when_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "MODEL_PATH", tmp_path / "missing.joblib")

    response = TestClient(api_main.app).post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert "Run the training pipeline" in response.json()["detail"]
