"""Portfolio Streamlit demo for the California Housing model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.mlapp.artifacts import (  # noqa: E402
    FEATURE_NAMES,
    InvalidModelArtifact,
    load_model_bundle,
    predict_from_bundle,
    validate_prediction,
)
from src.mlapp.data import UI_BOUNDS  # noqa: E402
from src.mlapp.pipeline import run_pipeline  # noqa: E402

MODEL_PATH = ROOT / "models" / "selected_model.joblib"
EXPORT_PATH = ROOT / "exports" / "metrics.json"

st.set_page_config(
    page_title="ML Prediction App — California Housing",
    layout="wide",
)

st.title("California Housing Historical Block-Group Estimator")
st.caption(
    "Illustrative historical median block-group target estimate using 1990 census data — "
    "not a property valuation or financial advice."
)

try:
    if not MODEL_PATH.is_file() or not EXPORT_PATH.is_file():
        with st.spinner("Preparing the local demonstration model..."):
            run_pipeline()

    bundle = load_model_bundle(MODEL_PATH)
    metrics = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    required_metrics = {"selection", "validation_metrics", "final_test_metrics"}
    if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
        raise ValueError("Metrics export has an invalid structure")
except (InvalidModelArtifact, OSError, ValueError, json.JSONDecodeError):
    st.error(
        "The demonstration model could not be prepared safely. "
        "Run `python -m src.mlapp.pipeline` and try again."
    )
    st.stop()

st.subheader("Model selection on validation data")
st.caption(
    "The fixed candidate models are compared on validation RMSE. "
    "The final test set is not used to choose the serving model."
)
left, right = st.columns(2)
with left:
    st.markdown("**Linear Regression baseline — validation**")
    st.json(metrics["validation_metrics"].get("linear_baseline", {}))
with right:
    st.markdown("**Random Forest — validation**")
    st.json(metrics["validation_metrics"].get("random_forest", {}))

selected_name = metrics["selection"].get("selected_model", "selected model")
st.subheader("Final held-out test evaluation")
st.markdown(f"**Selected model:** `{selected_name}`")
st.json(metrics["final_test_metrics"])
st.caption(
    "R² is the proportion of target variance explained on this specific held-out historical "
    "dataset split; it is not percent accuracy or confidence."
)

st.subheader("Selected-model feature importance")
importance = metrics.get("feature_importance_selected_model", [])
importance_frame = pd.DataFrame(importance)
if {"feature", "importance"}.issubset(importance_frame.columns):
    st.bar_chart(importance_frame.set_index("feature")["importance"])
    st.caption(
        "Impurity-based Random Forest importance is model inspection, not causal explanation, "
        "and can favour some feature structures."
    )
else:
    st.info("Feature-importance data is unavailable for the selected model.")

st.divider()
st.subheader("Try a historical block-group estimate")
st.caption(
    "The controls use convenience ranges for this demonstration. They are not the training-data "
    "schema or evidence of current market validity."
)

left, right = st.columns(2)
with left:
    med_inc = st.slider(
        "Median income (tens of thousands of $)",
        UI_BOUNDS["MedInc"][0],
        UI_BOUNDS["MedInc"][1],
        5.0,
        0.1,
    )
    house_age = st.slider(
        "Median house age (years)",
        int(UI_BOUNDS["HouseAge"][0]),
        int(UI_BOUNDS["HouseAge"][1]),
        20,
    )
    ave_rooms = st.slider(
        "Average rooms per household",
        UI_BOUNDS["AveRooms"][0],
        UI_BOUNDS["AveRooms"][1],
        6.0,
        0.1,
    )
    ave_bedrms = st.slider(
        "Average bedrooms per household",
        UI_BOUNDS["AveBedrms"][0],
        UI_BOUNDS["AveBedrms"][1],
        1.0,
        0.1,
    )
with right:
    population = st.slider(
        "Block group population",
        int(UI_BOUNDS["Population"][0]),
        int(UI_BOUNDS["Population"][1]),
        1_500,
    )
    ave_occup = st.slider(
        "Average household occupancy",
        UI_BOUNDS["AveOccup"][0],
        UI_BOUNDS["AveOccup"][1],
        3.0,
        0.1,
    )
    latitude = st.slider(
        "Latitude",
        UI_BOUNDS["Latitude"][0],
        UI_BOUNDS["Latitude"][1],
        34.0,
        0.1,
    )
    longitude = st.slider(
        "Longitude",
        UI_BOUNDS["Longitude"][0],
        UI_BOUNDS["Longitude"][1],
        -118.0,
        0.1,
    )

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
    prediction = validate_prediction(predict_from_bundle(bundle, input_frame)[0])
except (InvalidModelArtifact, IndexError, KeyError, TypeError, ValueError):
    st.error("The model returned an invalid estimate. Please retrain the demonstration model.")
else:
    st.metric("Estimated historical median block-group value", f"${prediction * 100_000:,.0f}")
    r2 = metrics["final_test_metrics"].get("r2")
    if isinstance(r2, int | float):
        st.caption(f"Selected-model final held-out R²: {r2:.2f}")

st.divider()
st.subheader("Limitations")
for note in metrics.get("limitations", []):
    if isinstance(note, str):
        st.markdown(f"- {note}")
