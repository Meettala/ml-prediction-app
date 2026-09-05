"""Data loading, validation and deterministic splitting for California Housing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
FEATURE_UNITS = {
    "MedInc": "tens of thousands of US dollars",
    "HouseAge": "years",
    "AveRooms": "rooms per household",
    "AveBedrms": "bedrooms per household",
    "Population": "people",
    "AveOccup": "people per household",
    "Latitude": "decimal degrees",
    "Longitude": "decimal degrees",
}

# Bounds below are serving/demo controls, not claims about the training-data schema.
# The third tuple value indicates whether the lower bound is exclusive.
API_BOUNDS = {
    "MedInc": (0.0, 20.0, False),
    "HouseAge": (0.0, 100.0, False),
    "AveRooms": (0.0, 30.0, True),
    "AveBedrms": (0.0, 10.0, True),
    "Population": (0.0, 100_000.0, False),
    "AveOccup": (0.0, 15.0, True),
    "Latitude": (32.0, 42.0, False),
    "Longitude": (-125.0, -114.0, False),
}

# Narrower controls chosen for a usable public Streamlit demonstration.
UI_BOUNDS = {
    "MedInc": (0.5, 15.0),
    "HouseAge": (1.0, 52.0),
    "AveRooms": (1.0, 15.0),
    "AveBedrms": (0.5, 5.0),
    "Population": (3.0, 10_000.0),
    "AveOccup": (0.5, 10.0),
    "Latitude": (32.5, 42.0),
    "Longitude": (-124.5, -114.0),
}

TARGET_NAME = "MedHouseVal"
REQUIRED_COLUMNS = (*FEATURE_DESCRIPTIONS, TARGET_NAME)

FINAL_TEST_SIZE = 0.20
VALIDATION_SIZE_OF_DEVELOPMENT = 0.25
FINAL_TEST_RANDOM_STATE = 42
VALIDATION_RANDOM_STATE = 43


@dataclass(frozen=True)
class CleaningAudit:
    """Exact row accounting for the fixed demonstration cleaning sequence."""

    raw_rows: int
    exact_duplicates_removed: int
    ave_rooms_rows_removed: int
    ave_occup_rows_removed: int
    final_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DataSplits:
    """Deterministic train/validation/final-test partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    def row_counts(self) -> dict[str, int]:
        return {
            "train": int(len(self.X_train)),
            "validation": int(len(self.X_validation)),
            "test": int(len(self.X_test)),
        }


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


def clean_data_with_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
    """Apply fixed transparent filters and return exact sequential row accounting."""
    validate_data_frame(frame)
    raw_rows = len(frame)

    deduplicated = frame.drop_duplicates()
    exact_duplicates_removed = raw_rows - len(deduplicated)

    rooms_filtered = deduplicated[deduplicated["AveRooms"] < 30]
    ave_rooms_rows_removed = len(deduplicated) - len(rooms_filtered)

    occupancy_filtered = rooms_filtered[rooms_filtered["AveOccup"] < 15]
    ave_occup_rows_removed = len(rooms_filtered) - len(occupancy_filtered)

    cleaned = occupancy_filtered.reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No housing rows remain after cleaning")

    audit = CleaningAudit(
        raw_rows=int(raw_rows),
        exact_duplicates_removed=int(exact_duplicates_removed),
        ave_rooms_rows_removed=int(ave_rooms_rows_removed),
        ave_occup_rows_removed=int(ave_occup_rows_removed),
        final_rows=int(len(cleaned)),
    )
    if (
        audit.raw_rows
        - audit.exact_duplicates_removed
        - audit.ave_rooms_rows_removed
        - audit.ave_occup_rows_removed
        != audit.final_rows
    ):
        raise RuntimeError("Cleaning row accounting is inconsistent")
    return cleaned, audit


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper returning only the cleaned frame."""
    cleaned, _ = clean_data_with_audit(frame)
    return cleaned


def _validate_fraction(value: float, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    fraction = float(value)
    if not 0 < fraction < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return fraction


def _validate_random_state(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def split_train_validation_test(
    frame: pd.DataFrame,
    final_test_size: float = FINAL_TEST_SIZE,
    validation_size_of_development: float = VALIDATION_SIZE_OF_DEVELOPMENT,
    final_test_random_state: int = FINAL_TEST_RANDOM_STATE,
    validation_random_state: int = VALIDATION_RANDOM_STATE,
) -> DataSplits:
    """Create deterministic train/validation/final-test partitions.

    The final test partition is created first and is not used for candidate
    model selection. Validation is then split from the remaining development
    rows and is used only for fixed-candidate comparison.
    """
    validate_data_frame(frame)
    test_fraction = _validate_fraction(final_test_size, "final_test_size")
    validation_fraction = _validate_fraction(
        validation_size_of_development,
        "validation_size_of_development",
    )
    test_state = _validate_random_state(final_test_random_state, "final_test_random_state")
    validation_state = _validate_random_state(
        validation_random_state,
        "validation_random_state",
    )

    features = frame[list(FEATURE_DESCRIPTIONS)]
    target = frame[TARGET_NAME]
    X_development, X_test, y_development, y_test = train_test_split(
        features,
        target,
        test_size=test_fraction,
        random_state=test_state,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=validation_fraction,
        random_state=validation_state,
    )

    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def split_data(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return the legacy deterministic two-way split for external compatibility.

    The production JR05 pipeline does not use this helper for model selection.
    """
    validate_data_frame(frame)
    fraction = _validate_fraction(test_size, "test_size")
    state = _validate_random_state(random_state, "random_state")
    features = frame[list(FEATURE_DESCRIPTIONS)]
    target = frame[TARGET_NAME]
    return train_test_split(features, target, test_size=fraction, random_state=state)
