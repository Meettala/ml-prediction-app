import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlapp.data import clean_data, load_data, split_data
from src.mlapp.train import train_linear_baseline, train_random_forest


def _split():
    df = clean_data(load_data())
    return split_data(df)


def test_random_forest_beats_reasonable_floor():
    X_train, X_test, y_train, y_test = _split()
    forest = train_random_forest(X_train, y_train, X_test, y_test)
    # Sanity floor, not a tight bound — catches a badly broken pipeline
    # without being brittle to minor library-version drift.
    assert forest.metrics["r2"] > 0.6


def test_linear_baseline_runs_and_reports_worse_or_equal_r2():
    X_train, X_test, y_train, y_test = _split()
    linear = train_linear_baseline(X_train, y_train, X_test, y_test)
    forest = train_random_forest(X_train, y_train, X_test, y_test)
    assert linear.metrics["r2"] <= forest.metrics["r2"] + 0.05
