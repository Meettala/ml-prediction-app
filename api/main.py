"""FastAPI service for the trusted local California Housing model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.mlapp.artifacts import (
    FEATURE_NAMES,
    InvalidModelArtifact,
    load_model_bundle,
    predict_from_bundle,
    validate_prediction,
)
from src.mlapp.data import API_BOUNDS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "selected_model.joblib"

app = FastAPI(
    title="California Housing Historical Block-Group Estimator",
    description=(
        "Illustrative historical median block-group target estimate from 1990 US Census data. "
        "Not a property valuation or financial advice."
    ),
)

_model_bundle: dict[str, Any] | None = None


class PredictionRequest(BaseModel):
    """Validated block-group features in canonical training-feature order."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    MedInc: float = Field(..., ge=API_BOUNDS["MedInc"][0], le=API_BOUNDS["MedInc"][1])
    HouseAge: float = Field(
        ...,
        ge=API_BOUNDS["HouseAge"][0],
        le=API_BOUNDS["HouseAge"][1],
    )
    AveRooms: float = Field(..., gt=API_BOUNDS["AveRooms"][0], le=API_BOUNDS["AveRooms"][1])
    AveBedrms: float = Field(
        ...,
        gt=API_BOUNDS["AveBedrms"][0],
        le=API_BOUNDS["AveBedrms"][1],
    )
    Population: float = Field(
        ...,
        ge=API_BOUNDS["Population"][0],
        le=API_BOUNDS["Population"][1],
    )
    AveOccup: float = Field(..., gt=API_BOUNDS["AveOccup"][0], le=API_BOUNDS["AveOccup"][1])
    Latitude: float = Field(..., ge=API_BOUNDS["Latitude"][0], le=API_BOUNDS["Latitude"][1])
    Longitude: float = Field(
        ...,
        ge=API_BOUNDS["Longitude"][0],
        le=API_BOUNDS["Longitude"][1],
    )


class PredictionResponse(BaseModel):
    predicted_value_100k: float
    predicted_value_usd: float
    model: str
    disclaimer: str


def get_model_bundle() -> dict[str, Any]:
    """Return the cached, validated trusted model bundle."""
    global _model_bundle
    if _model_bundle is None:
        try:
            _model_bundle = load_model_bundle(MODEL_PATH)
        except InvalidModelArtifact as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable. Run the training pipeline first.",
            ) from exc
    return _model_bundle


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_artifact_present": MODEL_PATH.is_file()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = get_model_bundle()
    values = request.model_dump()
    features = pd.DataFrame(
        [[values[name] for name in FEATURE_NAMES]],
        columns=list(FEATURE_NAMES),
    )

    try:
        raw_prediction = predict_from_bundle(bundle, features)[0]
        prediction = validate_prediction(raw_prediction)
    except (InvalidModelArtifact, IndexError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The prediction could not be produced safely.",
        ) from exc

    return PredictionResponse(
        predicted_value_100k=round(prediction, 3),
        predicted_value_usd=round(prediction * 100_000, 2),
        model=str(bundle["model_name"]),
        disclaimer=(
            "Illustrative historical median block-group estimate using 1990 census data; "
            "not a property valuation or financial advice."
        ),
    )
