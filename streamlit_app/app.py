"""Portfolio Streamlit demo for the California Housing model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.mlapp.artifacts import (
    FEATURE_NAMES,
    InvalidModelArtifact,
    load_model_bundle,
    validate_prediction,
)
from src.mlapp.pipeline import run_pipeline

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
EXPORT_PATH = ROOT / "exports" / "metrics.json"

st.set_page_config(
    page_title="ML Prediction App — California Housing",
    layout="wide",
)

st.title("California Housing Price Predictor")
st.caption(
    "Illustrative block-group estimate using 1990 census data — "
    "not a property valuation or financial advice."
)

try:
    if not MODEL_PATH.is_file() or not EXPORT_PATH.is_file():
        with st.spinner("Preparing the local demonstration model..."):
            run_pipeline()

    bundle = load_model_bundle(MODEL_PATH)
    metrics = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not isinstance(metrics.get("models"), dict):
        raise ValueError("Metrics export has an invalid structure")
except (InvalidModelArtifact, OSError, ValueError, json.JSONDecodeError):
    st.error(
        "The demonstration model could not be prepared safely. "
        "Run `python -m src.mlapp.pipeline` and try again."
    )
    st.stop()

st.subheader("Held-out test evaluation")
left, right = st.columns(2)
with left:
    st.markdown("**Random Forest** — used for the estimate below")
    st.json(metrics["models"].get("random_forest", {}))
with right:
    st.markdown("**Linear baseline** — shown for honest comparison")
    st.json(metrics["models"].get("linear_baseline", {}))

st.subheader("Feature importance")
importance = metrics.get("feature_importance_random_forest", [])
importance_frame = pd.DataFrame(importance)
if {"feature", "importance"}.issubset(importance_frame.columns):
    st.bar_chart(importance_frame.set_index("feature")["importance"])
else:
    st.info("Feature-importance data is unavailable in this metrics export.")

st.divider()
st.subheader("Try a block-group estimate")
st.caption("Inputs are constrained to plausible ranges used by this demonstration.")

left, right = st.columns(2)
with left:
    med_inc = st.slider("Median income (tens of thousands of $)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("Median house age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average rooms per household", 1.0, 15.0, 6.0, 0.1)
    ave_bedrms = st.slider("Average bedrooms per household", 0.5, 5.0, 1.0, 0.1)
with right:
    population = st.slider("Block group population", 3, 10_000, 1_500)
    ave_occup = st.slider("Average household occupancy", 0.5, 10.0, 3.0, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -124.5, -114.0, -118.0, 0.1)

values = {
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
}
input_frame = pd.DataFrame(
    [[values[name] for name in FEATURE_NAMES]],
    columns=list(FEATURE_NAMES),
)

try:
    prediction = validate_prediction(bundle["model"].predict(input_frame)[0])
except (InvalidModelArtifact, IndexError, KeyError, TypeError, ValueError):
    st.error("The model returned an invalid estimate. Please retrain the demonstration model.")
else:
    st.metric("Estimated median block-group value", f"${prediction * 100_000:,.0f}")
    random_forest_metrics = metrics["models"].get("random_forest", {})
    r2 = random_forest_metrics.get("r2")
    if isinstance(r2, int | float):
        st.caption(f"Random Forest held-out R²: {r2:.2f}")

st.divider()
st.subheader("Limitations")
for note in metrics.get("limitations", []):
    if isinstance(note, str):
        st.markdown(f"- {note}")
