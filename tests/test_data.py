import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlapp.data import clean_data, load_data, split_data


def test_load_data_has_expected_columns():
    df = load_data()
    assert "MedHouseVal" in df.columns
    assert "MedInc" in df.columns
    assert len(df) > 1000


def test_clean_data_removes_extreme_outliers():
    df = load_data()
    cleaned = clean_data(df)
    assert cleaned["AveRooms"].max() < 30
    assert cleaned["AveOccup"].max() < 15
    assert len(cleaned) <= len(df)


def test_split_data_shapes_match():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
