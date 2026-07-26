from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import src.mlapp.data as data_module
from src.mlapp.data import FEATURE_DESCRIPTIONS, TARGET_NAME, clean_data, load_data, split_data


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [5.0, 20, 6.0, 1.0, 1500, 3.0, 34.0, -118.0, 2.5],
            [4.0, 30, 5.0, 1.0, 900, 2.5, 36.0, -120.0, 2.0],
            [8.0, 10, 35.0, 1.2, 1100, 2.0, 37.0, -121.0, 3.5],
            [3.0, 40, 4.0, 1.0, 2000, 20.0, 38.0, -122.0, 1.5],
            [6.0, 15, 7.0, 1.1, 1200, 2.8, 35.0, -119.0, 3.0],
        ],
        columns=[*FEATURE_DESCRIPTIONS, TARGET_NAME],
    )


def test_load_data_returns_a_copy_with_expected_schema(monkeypatch):
    original = sample_frame()
    monkeypatch.setattr(
        data_module,
        "fetch_california_housing",
        lambda as_frame: SimpleNamespace(frame=original),
    )

    loaded = load_data()

    assert list(loaded.columns) == [*FEATURE_DESCRIPTIONS, TARGET_NAME]
    assert loaded is not original


def test_clean_data_removes_duplicates_and_extreme_rows():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)

    cleaned = clean_data(frame)

    assert cleaned["AveRooms"].max() < 30
    assert cleaned["AveOccup"].max() < 15
    assert len(cleaned) == 3


def test_split_data_is_reproducible_and_preserves_alignment():
    frame = pd.concat([sample_frame().iloc[[0, 1, 4]]] * 10, ignore_index=True)

    first = split_data(frame, test_size=0.2, random_state=42)
    second = split_data(frame, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = first

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.index.tolist() == second[0].index.tolist()
    assert X_test.index.tolist() == second[1].index.tolist()
