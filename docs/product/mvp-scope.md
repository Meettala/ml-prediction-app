# MVP scope — ML Prediction App

## In scope

- End-to-end pipeline: load, validate, audibly clean, create train/validation/final-test splits, fit two fixed candidate models, select on validation RMSE, refit the selected model and evaluate once on final test.
- Honest baseline comparison: Linear Regression versus Random Forest.
- FastAPI serving endpoint with strict input validation and safe artifact/output failures.
- Streamlit interactive demo using the real selected model.
- Generated machine-readable evidence separating validation selection from final held-out metrics.
- Historical median block-group value estimates with explicit non-valuation limitations.

## Explicitly out of scope

- Hyperparameter-search platforms or adding model families merely to improve a portfolio score.
- Real-time/current housing data; the dataset is a fixed historical 1990 census snapshot.
- Individual property appraisal or valuation.
- Financial or investment guidance.
- Causal interpretation of Random Forest feature importance.
- Production authentication, model registry, drift monitoring or governance infrastructure.

The project optimizes for a small, reproducible and defensible ML lifecycle rather than maximum benchmark performance.
