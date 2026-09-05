# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Deterministic train/validation/final-test evaluation with final test held out from model selection.
- Auditable cleaning row counts and explicit split purposes in generated metrics.
- Exact validated training-environment metadata in `exports/metrics.json`.
- Selected-model artifact bundle supporting the fitted preprocessing scaler when required.
- Exact scikit-learn artifact/runtime compatibility enforcement for trusted local artifacts.
- Shared feature descriptions, units, API demo bounds and Streamlit convenience bounds.
- End-to-end deterministic pipeline integration coverage.
- `pip check`, real-data training/reproducibility evidence and Docker API smoke gates in CI.
- CI comparison of committed metrics against a fresh real California Housing pipeline run.

### Changed

- Fixed model-selection methodology: Linear Regression and Random Forest are compared on validation RMSE, not final-test results.
- Random Forest is selected from validation evidence, refit on train + validation and evaluated once on final test.
- Reconciled the earlier README/metrics contradiction with fresh generated evidence: final selected-model test RMSE 0.4771, MAE 0.3202 and R² 0.8290 in the validated JR05 environment.
- Aligned `requirements.txt` dependency bounds with `pyproject.toml`.
- FastAPI and Streamlit now serve `models/selected_model.joblib` and use shared feature metadata.
- Public wording now emphasizes a historical median block-group estimate rather than a current property-price prediction.
- Documentation distinguishes validation metrics, final-test metrics, R² interpretation and non-causal feature importance.
- Historical deployment URLs are qualified by their last manual verification date until current JR05 smoke testing is completed.

## [0.1.0] - 2026-07-20

### Added

- Initial California Housing data pipeline.
- Linear Regression and Random Forest training.
- RMSE, MAE and R² reporting.
- FastAPI prediction service.
- Streamlit demonstration.
