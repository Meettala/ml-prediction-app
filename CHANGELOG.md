# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Versioned, validated Random Forest model bundle.
- Canonical feature ordering shared by training, FastAPI and Streamlit.
- Offline artifact, API and data tests.
- Python 3.10–3.12 CI, Ruff and dependency auditing.
- Non-root Docker API image with a health check.
- MIT licence, security policy, contribution guide and living AI handoff.

### Changed

- FastAPI now rejects extra, non-finite and out-of-range inputs.
- Invalid model output and missing/corrupt artifacts return safe errors.
- Streamlit now loads the validated bundle and handles invalid metrics safely.
- Unit tests no longer download or train the real dataset during import.
- Training output paths are configurable for isolated tests.

## [0.1.0] - 2026-07-20

### Added

- Initial California Housing data pipeline.
- Linear Regression and Random Forest training.
- RMSE, MAE and R² reporting.
- FastAPI prediction service.
- Streamlit demonstration.
