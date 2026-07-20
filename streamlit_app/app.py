"""
California Housing Price Predictor — Streamlit demo.

Run locally with:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mlapp.data import FEATURE_DESCRIPTIONS  # noqa: E402
from src.mlapp.pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="ML Prediction App — California Housing", layout="wide")

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
EXPORT_PATH = ROOT / "exports" / "metrics.json"

st.title("California Housing Price Predictor")
st.caption(
    "Illustrative portfolio project — 1990 census data, not financial advice."
)

if not MODEL_PATH.exists():
    with st.spinner("Training model for the first time..."):
        run_pipeline()

model = joblib.load(MODEL_PATH)
import json
metrics = json.loads(EXPORT_PATH.read_text())

st.subheader("Model evaluation (held-out test set)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Random Forest** (used for predictions below)")
    st.json(metrics["models"]["random_forest"])
with c2:
    st.markdown("**Linear baseline** (for honest comparison)")
    st.json(metrics["models"]["linear_baseline"])

st.subheader("Feature importance")
fi_df = pd.DataFrame(metrics["feature_importance_random_forest"])
st.bar_chart(fi_df.set_index("feature")["importance"])

st.divider()
st.subheader("Try a prediction")
st.caption("Enter block-group characteristics to estimate median house value.")

col1, col2 = st.columns(2)
with col1:
    med_inc = st.slider("Median income (tens of thousands $)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("Median house age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average rooms per household", 1.0, 15.0, 6.0, 0.1)
    ave_bedrms = st.slider("Average bedrooms per household", 0.5, 5.0, 1.0, 0.1)
with col2:
    population = st.slider("Block group population", 3, 10000, 1500)
    ave_occup = st.slider("Average household occupancy", 0.5, 10.0, 3.0, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -124.5, -114.0, -118.0, 0.1)

input_df = pd.DataFrame([{
    "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
    "Latitude": latitude, "Longitude": longitude,
}])

prediction = model.predict(input_df)[0]
st.metric("Predicted median house value", f"${prediction * 100_000:,.0f}")
st.caption("Random Forest model, R² ≈ {:.2f} on held-out test data.".format(
    metrics["models"]["random_forest"]["r2"]
))

st.divider()
st.subheader("Limitations")
for note in metrics["limitations"]:
    st.markdown(f"- {note}")
