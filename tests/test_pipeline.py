from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.mlapp.pipeline as pipeline_module
from src.mlapp.artifacts import load_model_bundle
from src.mlapp.data import FEATURE_DESCRIPTIONS, TARGET_NAME


def synthetic_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "MedInc": rng.uniform(1.0, 10.0, rows),
            "HouseAge": rng.integers(1, 53, rows).astype(float),
            "AveRooms": rng.uniform(2.0, 12.0, rows),
            "AveBedrms": rng.uniform(0.5, 3.0, rows),
            "Population": rng.integers(100, 5000, rows).astype(float),
            "AveOccup": rng.uniform(1.0, 8.0, rows),
            "Latitude": rng.uniform(32.5, 42.0, rows),
            "Longitude": rng.uniform(-124.5, -114.0, rows),
        }
    )
    frame[TARGET_NAME] = (
        0.35 * frame["MedInc"]
        + 0.01 * frame["HouseAge"]
        - 0.08 * frame["AveOccup"]
        + rng.normal(0, 0.03, rows)
    )
    assert list(frame.columns) == [*FEATURE_DESCRIPTIONS, TARGET_NAME]
    return frame


def test_pipeline_is_deterministic_and_keeps_final_test_out_of_selection(monkeypatch, tmp_path):
    frame = synthetic_frame()
    monkeypatch.setattr(pipeline_module, "load_data", lambda: frame.copy())

    first_models = tmp_path / "models-one"
    first_metrics = tmp_path / "metrics-one.json"
    second_models = tmp_path / "models-two"
    second_metrics = tmp_path / "metrics-two.json"

    first = pipeline_module.run_pipeline(first_models, first_metrics)
    second = pipeline_module.run_pipeline(second_models, second_metrics)

    assert first == second
    assert first_metrics.read_bytes() == second_metrics.read_bytes()
    assert json.loads(first_metrics.read_text(encoding="utf-8")) == first
    assert first["schema_version"] == 2
    assert first["cleaning"]["raw_rows"] == 120
    assert first["cleaning"]["final_rows"] == 120
    assert sum(first["split"]["row_counts"].values()) == 120
    assert first["selection"]["final_test_used_for_selection"] is False
    assert set(first["validation_metrics"]) == {"linear_baseline", "random_forest"}
    assert first["final_test_metrics"]["model"] == first["selection"]["selected_model"]

    bundle = load_model_bundle(first_models / pipeline_module.SELECTED_MODEL_FILENAME)
    assert bundle["model_name"] == first["selection"]["selected_model"]
