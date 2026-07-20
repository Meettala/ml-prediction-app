"""
FastAPI service serving the trained Random Forest model.

Run locally with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

app = FastAPI(
    title="California Housing Price Predictor",
    description=(
        "Illustrative portfolio project predicting median block-group "
        "house value from 1990 US Census data. Not financial advice — "
        "see /docs/security/safety-rules.md in the repo."
    ),
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python -m src.mlapp.pipeline` first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income (tens of thousands of $)", ge=0)
    HouseAge: float = Field(..., description="Median house age (years)", ge=0, le=100)
    AveRooms: float = Field(..., description="Average rooms per household", gt=0)
    AveBedrms: float = Field(..., description="Average bedrooms per household", gt=0)
    Population: float = Field(..., description="Block group population", ge=0)
    AveOccup: float = Field(..., description="Average household occupancy", gt=0)
    Latitude: float = Field(..., description="Latitude", ge=32.0, le=42.0)
    Longitude: float = Field(..., description="Longitude", ge=-125.0, le=-114.0)


class PredictionResponse(BaseModel):
    predicted_value_100k: float
    predicted_value_usd: float
    model: str
    disclaimer: str


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()
    features = pd.DataFrame([{
        "MedInc": request.MedInc, "HouseAge": request.HouseAge,
        "AveRooms": request.AveRooms, "AveBedrms": request.AveBedrms,
        "Population": request.Population, "AveOccup": request.AveOccup,
        "Latitude": request.Latitude, "Longitude": request.Longitude,
    }])
    prediction = float(model.predict(features)[0])
    return PredictionResponse(
        predicted_value_100k=round(prediction, 3),
        predicted_value_usd=round(prediction * 100_000, 2),
        model="random_forest",
        disclaimer="Illustrative estimate on 1990 census data — not financial advice.",
    )
