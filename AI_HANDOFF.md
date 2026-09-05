# AI Handoff — ML Prediction App

> Continue from the live repository and current CI, not from old portfolio instructions. Do not start final portfolio/JR06 consolidation unless the user explicitly requests it.

## Repository

- Repository: `Meettala/ml-prediction-app`
- Default branch: `main`
- Visibility: public
- Licence: MIT
- Package version: `0.1.0`
- Stack: Python, pandas, scikit-learn, FastAPI, Streamlit, Pydantic, pytest, Ruff and Docker
- Dataset: scikit-learn California Housing historical block-group dataset

## JR05 evaluation truth

The final-test set is **not** used to choose the serving model.

Deterministic design:

1. clean the real California Housing data with fixed transparent filters;
2. create a 20% final test split first (`random_state=42`);
3. split the remaining development rows into train/validation, using 25% of development for validation (`random_state=43`);
4. fit the fixed Linear Regression and Random Forest candidates on train;
5. select by lowest validation RMSE; exact tie → Linear Regression baseline;
6. refit the selected model on train + validation;
7. evaluate once on final test.

Current generated evidence:

- raw rows: 20,640
- duplicates removed: 0
- `AveRooms >= 30` removed: 24
- `AveOccup >= 15` removed after prior filtering: 19
- cleaned rows: 20,597
- train: 12,357
- validation: 4,120
- final test: 4,120
- development refit rows: 16,477

Validation metrics:

- Linear Regression: RMSE 0.6877, MAE 0.5023, R² 0.6454
- Random Forest: RMSE 0.5356, MAE 0.3529, R² 0.7850

Selected model: **Random Forest**.

Final held-out test metrics after refit:

- RMSE 0.4771
- MAE 0.3202
- R² 0.8290

`exports/metrics.json` is the authoritative generated evidence. CI regenerates it from the real dataset, checks it against the committed file, then runs the pipeline again and requires byte-for-byte reproducibility in the same validated environment.

## Validated JR05 environment

- Python 3.12.14
- pandas 2.3.3
- NumPy 2.5.2
- scikit-learn 1.9.0
- joblib 1.6.0

`requirements.txt` and `pyproject.toml` use aligned compatible ranges. Do not claim bit-for-bit reproducibility across arbitrary dependency versions.

## Data and target boundary

The dataset is derived from 1990 US Census block-group aggregates. The target is median block-group house value in $100,000 dataset units.

Public wording should prefer **historical median block-group value estimate**.

Do not call the output:

- current property value;
- house appraisal;
- investment value;
- current California housing price;
- financial advice.

R² 0.8290 is variance explained on one historical held-out split. It is not 82.9% prediction accuracy or confidence.

## Cleaning truth

The thresholds `AveRooms < 30` and `AveOccup < 15` are fixed transparent demonstration filters for extreme aggregate values. They are not claimed to be statistically optimal outlier treatment.

## Feature/schema truth

Canonical feature order:

1. `MedInc`
2. `HouseAge`
3. `AveRooms`
4. `AveBedrms`
5. `Population`
6. `AveOccup`
7. `Latitude`
8. `Longitude`

Shared metadata in `src/mlapp/data.py` distinguishes:

- feature description/unit;
- cleaned training-data range;
- API demo/safety range;
- Streamlit convenience range.

Do not call the UI/API bounds the training schema.

## Artifact trust boundary

The serving artifact is `models/selected_model.joblib`, built by the trusted training pipeline.

The artifact stores:

- schema version;
- selected model name;
- canonical features/order;
- exact scikit-learn version;
- fitted estimator;
- fitted scaler if the selected model requires one.

The loader requires the recorded scikit-learn version to exactly match the runtime version. On mismatch it asks for retraining.

This does **not** make arbitrary joblib/pickle files safe. They can execute code during loading. Never add model upload or a user-controlled artifact path.

## API safety model

FastAPI/Pydantic:

- forbids extra fields;
- rejects missing fields;
- rejects non-finite values;
- applies explicit demo/safety ranges;
- creates the prediction frame in canonical feature order;
- validates model output is numeric, finite and non-negative;
- returns generic errors for missing/corrupt/incompatible artifacts and invalid output;
- does not return local paths or raw deserialization errors.

## Feature importance boundary

Random Forest impurity importance is model inspection, not causal explanation, and can favour some feature structures. Do not present it as a causal or policy result.

## Current automated verification

JR05 CI contains:

- tests on Python 3.10, 3.11 and 3.12;
- `pip check`;
- Ruff;
- `pip-audit`;
- a real California Housing training-evidence job;
- committed-metrics comparison;
- a second real pipeline run for reproducibility;
- a Docker build from scratch;
- non-root runtime-user check;
- `/health` and OpenAPI smoke;
- one valid `/predict` and one invalid request.

Always verify the latest exact `main` CI before repeating these as current claims.

## Historical deployment URLs

Last manually verified on 1 August 2026:

- Streamlit: `https://meet-tala-ml-prediction-app-px2pch6zms5lrdhxyxurht.streamlit.app`
- FastAPI: `https://ml-prediction-app-xfsd.onrender.com`
- API docs: `https://ml-prediction-app-xfsd.onrender.com/docs`
- health: `https://ml-prediction-app-xfsd.onrender.com/health`

JR05 automation/browser fetch failure is not evidence the deployments are down. Until a current JR05 deployment is freshly smoke-tested, describe these as historical deployment candidates rather than current verified deployments.

## Known limitations

- historical 1990 dataset;
- block-group aggregates, not individual properties;
- one deterministic final test split, not temporal validation;
- no current-market or fairness validation;
- no uncertainty interval;
- feature importance is non-causal;
- public serving architecture has no authentication, rate limiting, drift monitoring or governed registry;
- current hosted JR05 deployment status requires fresh manual verification if automated access remains unavailable.

## Genuine remaining presentation conditions

- Freshly smoke the historical Streamlit and Render deployment candidates after JR05 is deployed, if they are still used.
- Capture a clean current screenshot/video only after the deployed content matches `main`.
- Add repository description/topics manually if repository-settings tooling is unavailable.

Do not resume old project-order tasks or the stale instruction to merge an already-merged documentation PR.

## Rules for another AI

Inspect current repository/CI/deployments first. Preserve train/validation/final-test separation and the untouched-test claim. Do not use the final test to tune/select models. Keep `exports/metrics.json` generated and evidence-backed. Treat joblib as trusted code only. Add focused positive and negative tests for material behaviour changes. Keep prediction, R², feature-importance and historical-data claims conservative.
