import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure a trained model exists before importing the API (which loads it lazily,
# but /predict needs it present to succeed).
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "models" / "random_forest.joblib").exists():
    subprocess.run([sys.executable, "-m", "src.mlapp.pipeline"], cwd=ROOT, check=True)

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "MedInc": 5.0, "HouseAge": 20, "AveRooms": 6.0, "AveBedrms": 1.0,
    "Population": 1500, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_trained"] is True


def test_predict_returns_reasonable_value():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_value_100k"] > 0
    # usd is derived from the unrounded prediction, so allow for the
    # small rounding difference vs. the already-rounded 100k figure.
    assert abs(body["predicted_value_usd"] - body["predicted_value_100k"] * 100_000) < 100
    assert "disclaimer" in body


def test_predict_rejects_invalid_input():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["HouseAge"] = -5  # invalid: below allowed range
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
