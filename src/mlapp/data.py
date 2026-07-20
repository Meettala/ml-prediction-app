"""
Data loading and preparation for the California Housing prediction task.

Dataset: scikit-learn's built-in California Housing dataset (derived from
the 1990 US Census, block-group level, no personal data — see
docs/security/privacy-by-design.md). Target: median house value, in units
of $100,000s, for a census block group.
"""

from __future__ import annotations

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

TARGET_NAME = "MedHouseVal"  # median house value, in $100,000s


def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic, transparent cleaning: drop exact duplicate rows and clip a
    small number of extreme outliers in AveRooms/AveOccup that are known
    data artifacts in this dataset (not real households), rather than
    silently leaving them to distort the model.
    """
    df = df.drop_duplicates()
    df = df[df["AveRooms"] < 30]
    df = df[df["AveOccup"] < 15]
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[list(FEATURE_DESCRIPTIONS.keys())]
    y = df[TARGET_NAME]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
