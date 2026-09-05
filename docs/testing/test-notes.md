# Testing notes — ML Prediction App

JR05 separates fast deterministic unit/integration tests from the real-data training-evidence job.

- `tests/test_data.py` — schema validation, exact cleaning row accounting, deterministic/disjoint train-validation-test splits, invalid split arguments and shared API/UI feature-bound metadata.
- `tests/test_train.py` — fixed Linear Regression/Random Forest candidates on synthetic data, finite metrics, exact Random Forest parameters, validation-RMSE selection and the simpler-model tie-breaker.
- `tests/test_artifacts.py` — versioned trusted bundles, preprocessing scaler path, missing/unversioned/schema-mismatch artifacts, exact scikit-learn compatibility rejection and invalid model outputs.
- `tests/test_api.py` — `/health`, OpenAPI, canonical feature ordering, valid historical estimate semantics, missing/extra/non-finite/out-of-range input rejection, safe invalid-output errors and safe missing/corrupt/incompatible-artifact errors.
- `tests/test_pipeline.py` — synthetic end-to-end determinism, final-test-not-used-for-selection invariant, generated JSON equality and selected-artifact loading.

CI additionally runs the **real scikit-learn California Housing pipeline** on Python 3.12, requires the fresh generated metrics file to match the committed `exports/metrics.json`, then runs the pipeline a second time and requires byte-for-byte reproduction in the same environment.

Docker CI builds from scratch, verifies non-root `appuser`, `/health`, OpenAPI, one valid prediction and one rejected invalid request.

Run locally:

```bash
python -m pip check
python -m pytest
ruff check src api streamlit_app tests tools
pip-audit -r requirements.txt
python -m src.mlapp.pipeline
python -m tools.verify_reproducibility
```

Quote a raw test count only from the latest executed CI rather than treating this note as a frozen count source.
