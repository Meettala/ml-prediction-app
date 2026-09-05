from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

import src.mlapp.data as data_module
from src.mlapp.data import (
    API_BOUNDS,
    FEATURE_DESCRIPTIONS,
    TARGET_NAME,
    UI_BOUNDS,
    clean_data,
    clean_data_with_audit,
    load_data,
    split_data,
    split_train_validation_test,
    validate_data_frame,
)


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


def test_clean_data_records_exact_sequential_row_accounting():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)

    cleaned, audit = clean_data_with_audit(frame)

    assert len(cleaned) == 3
    assert audit.to_dict() == {
        "raw_rows": 6,
        "exact_duplicates_removed": 1,
        "ave_rooms_rows_removed": 1,
        "ave_occup_rows_removed": 1,
        "final_rows": 3,
    }
    assert len(clean_data(frame)) == audit.final_rows


def test_three_way_split_is_reproducible_disjoint_and_complete():
    frame = pd.concat([sample_frame().iloc[[0, 1, 4]]] * 20, ignore_index=True)

    first = split_train_validation_test(frame)
    second = split_train_validation_test(frame)

    assert first.row_counts() == {"train": 36, "validation": 12, "test": 12}
    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_validation.index.tolist() == second.X_validation.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()

    train_indexes = set(first.X_train.index)
    validation_indexes = set(first.X_validation.index)
    test_indexes = set(first.X_test.index)
    assert train_indexes.isdisjoint(validation_indexes)
    assert train_indexes.isdisjoint(test_indexes)
    assert validation_indexes.isdisjoint(test_indexes)
    assert len(train_indexes | validation_indexes | test_indexes) == len(frame)


def test_serving_and_ui_bounds_share_the_canonical_feature_keys():
    expected = list(FEATURE_DESCRIPTIONS)
    assert list(API_BOUNDS) == expected
    assert list(UI_BOUNDS) == expected
    for feature in expected:
        api_min, api_max, _ = API_BOUNDS[feature]
        ui_min, ui_max = UI_BOUNDS[feature]
        assert api_min <= ui_min <= ui_max <= api_max


def test_legacy_two_way_split_remains_reproducible():
    frame = pd.concat([sample_frame().iloc[[0, 1, 4]]] * 10, ignore_index=True)
    first = split_data(frame, test_size=0.2, random_state=42)
    second = split_data(frame, test_size=0.2, random_state=42)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[1].index.tolist() == second[1].index.tolist()


def test_rejects_missing_non_numeric_and_non_finite_columns():
    missing = sample_frame().drop(columns=["MedInc"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_data_frame(missing)

    non_numeric = sample_frame()
    non_numeric["MedInc"] = "unknown"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_data_frame(non_numeric)

    non_finite = sample_frame()
    non_finite.loc[0, "MedInc"] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_data_frame(non_finite)


def test_rejects_empty_cleaning_result():
    frame = sample_frame().iloc[[2, 3]]
    with pytest.raises(ValueError, match="No housing rows remain"):
        clean_data(frame)


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, "0.2", True])
def test_rejects_invalid_three_way_split_fractions(value):
    with pytest.raises(ValueError):
        split_train_validation_test(sample_frame(), final_test_size=value)
    with pytest.raises(ValueError):
        split_train_validation_test(sample_frame(), validation_size_of_development=value)


@pytest.mark.parametrize("value", [1.5, "42", True])
def test_rejects_invalid_three_way_random_states(value):
    with pytest.raises(ValueError):
        split_train_validation_test(sample_frame(), final_test_random_state=value)
    with pytest.raises(ValueError):
        split_train_validation_test(sample_frame(), validation_random_state=value)
