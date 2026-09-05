from __future__ import annotations

import math

from fastapi.testclient import TestClient

import api.main as api_main
from src.mlapp.artifacts import FEATURE_NAMES, InvalidModelArtifact


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


def test_health_and_openapi_are_available():
    client = TestClient(api_main.app)
    health = client.get("/health")
    openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert isinstance(health.json()["model_artifact_present"], bool)
    assert openapi.status_code == 200
    assert "/predict" in openapi.json()["paths"]


def test_predict_uses_canonical_feature_order_and_historical_semantics():
    model = RecordingModel()
    api_main._model_bundle = {
        "model": model,
        "model_name": "random_forest",
        "scaler": None,
    }

    response = TestClient(api_main.app).post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert model.columns == list(FEATURE_NAMES)
    assert response.json()["predicted_value_usd"] == 250000.0
    assert response.json()["model"] == "random_forest"
    assert "historical" in response.json()["disclaimer"].lower()
    assert "not a property valuation" in response.json()["disclaimer"].lower()


def test_rejects_missing_extra_non_finite_and_out_of_range_inputs():
    client = TestClient(api_main.app)

    missing = {key: value for key, value in VALID_PAYLOAD.items() if key != "MedInc"}
    assert client.post("/predict", json=missing).status_code == 422
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
        "scaler": None,
    }

    response = TestClient(api_main.app).post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["detail"] == "The prediction could not be produced safely."


def test_returns_safe_503_for_missing_corrupt_or_incompatible_artifact(monkeypatch, tmp_path):
    client = TestClient(api_main.app)

    missing_path = tmp_path / "missing.joblib"
    monkeypatch.setattr(api_main, "MODEL_PATH", missing_path)
    missing_response = client.post("/predict", json=VALID_PAYLOAD)
    assert missing_response.status_code == 503
    assert str(tmp_path) not in missing_response.text

    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a joblib artifact")
    api_main._model_bundle = None
    monkeypatch.setattr(api_main, "MODEL_PATH", corrupt_path)
    corrupt_response = client.post("/predict", json=VALID_PAYLOAD)
    assert corrupt_response.status_code == 503
    assert str(tmp_path) not in corrupt_response.text

    api_main._model_bundle = None
    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda path: (_ for _ in ()).throw(
            InvalidModelArtifact("different scikit-learn version: local/details")
        ),
    )
    incompatible_response = client.post("/predict", json=VALID_PAYLOAD)
    assert incompatible_response.status_code == 503
    assert "scikit" not in incompatible_response.text.lower()
    assert "local/details" not in incompatible_response.text
