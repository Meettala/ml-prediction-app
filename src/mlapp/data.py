"""Data loading and preparation for the California Housing prediction task."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURE_DESCRIPTIONS = {
    "MedInc": "Median income in the block group (tens of thousands of $)",
    "HouseAge": "Median house age in the block group (years)",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average number of household members",
    "Latitude": "Block group latitude",
    "Longitude": "Block group longitude",
}
TARGET_NAME = "MedHouseVal"
REQUIRED_COLUMNS = (*FEATURE_DESCRIPTIONS, TARGET_NAME)


def load_data() -> pd.DataFrame:
    """Download the public aggregate dataset and return a validated copy."""
    data = fetch_california_housing(as_frame=True)
    frame = data.frame.copy()
    validate_data_frame(frame)
    return frame


def validate_data_frame(frame: pd.DataFrame) -> None:
    """Validate the training schema without silently coercing bad values."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Housing data must be a non-empty DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Housing data is missing required columns: {', '.join(missing)}")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Housing column {column} must be numeric")
        if frame[column].isna().any():
            raise ValueError(f"Housing column {column} contains missing values")
        if not frame[column].map(lambda value: math.isfinite(float(value))).all():
            raise ValueError(f"Housing column {column} contains non-finite values")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent duplicate and known extreme-artifact filtering."""
    validate_data_frame(frame)
    cleaned = frame.drop_duplicates()
    cleaned = cleaned[cleaned["AveRooms"] < 30]
    cleaned = cleaned[cleaned["AveOccup"] < 15]
    cleaned = cleaned.reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No housing rows remain after cleaning")
    return cleaned


def split_data(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return a deterministic train/test split after validating parameters."""
    validate_data_frame(frame)
    if not isinstance(test_size, int | float) or isinstance(test_size, bool):
        raise ValueError("test_size must be numeric")
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer")

    features = frame[list(FEATURE_DESCRIPTIONS)]
    target = frame[TARGET_NAME]
    return train_test_split(
        features,
        target,
        test_size=float(test_size),
        random_state=random_state,
    )
