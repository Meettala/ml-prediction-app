# Portfolio Presentation Guide

This guide covers manual presentation and deployment checks that require a running application or GitHub repository settings.

## 1. Run and verify the real training pipeline

```bash
git clone https://github.com/Meettala/ml-prediction-app.git
cd ml-prediction-app
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m src.mlapp.pipeline
python -m tools.verify_reproducibility
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The first real run downloads the public California Housing dataset and writes trusted local outputs to `models/selected_model.joblib` and `exports/metrics.json`. The second run verifies that the same validated environment reproduces the generated metrics byte-for-byte.

## 2. Explain the model evidence correctly

The recruiter story is:

1. historical California Housing block-group dataset;
2. auditable fixed cleaning;
3. final test split created before model comparison;
4. Linear Regression baseline and Random Forest compared on validation RMSE;
5. Random Forest selected on validation evidence;
6. selected model refit on train + validation;
7. one final held-out test evaluation: RMSE 0.4771, MAE 0.3202, R² 0.8290.

Do not call R² 0.8290 “82.9% accuracy.” Describe it as variance explained on this specific historical held-out split.

## 3. Run the demos

FastAPI:

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`; verify `/health`, a valid `/predict`, a missing-field request and an out-of-range request.

Streamlit:

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501` and confirm validation comparison and final-test evidence are shown separately.

Docker API:

```bash
docker build --no-cache -t ml-prediction-app .
docker run --rm -p 8000:8000 ml-prediction-app
```

## 4. Historical hosted deployment smoke checklist

Historical candidates, last manually verified on 1 August 2026:

- Streamlit: `https://meet-tala-ml-prediction-app-px2pch6zms5lrdhxyxurht.streamlit.app`
- API: `https://ml-prediction-app-xfsd.onrender.com`
- docs: `https://ml-prediction-app-xfsd.onrender.com/docs`
- health: `https://ml-prediction-app-xfsd.onrender.com/health`

After JR05 is deployed, manually verify:

### Streamlit

- page loads without a traceback;
- historical/non-valuation disclaimer is visible;
- validation metrics match the committed JR05 evidence;
- selected model is Random Forest;
- final test metrics are 0.4771 RMSE, 0.3202 MAE and 0.8290 R²;
- feature importance renders;
- changing an input changes the estimate;
- limitations remain visible.

### FastAPI

- `/health` returns `status=ok` and confirms the artifact is present;
- `/docs` or `/openapi.json` responds;
- valid `/predict` returns model `random_forest`, a finite estimate and historical disclaimer;
- missing-field and out-of-range requests are rejected;
- no local path or raw deserialization error is exposed.

A failed automation fetch alone is not proof that a free hosted deployment is down.

## 5. Capture portfolio media

Capture one clean current Streamlit screenshot only after it matches JR05 `main`. Show:

- historical/non-valuation disclaimer;
- validation model comparison;
- selected model and final held-out metrics;
- feature-importance chart;
- input controls and one example estimate;
- limitations.

Optional second image: FastAPI docs or one valid `/predict` response with model name/disclaimer.

Do not include browser tabs with personal information, tokens, local filesystem paths, environment variables or private data.

## 6. GitHub social preview and metadata

Convert `docs/assets/social-preview.svg` to a 1280×640 PNG if desired, then upload it through GitHub repository Settings.

Recommended description:

> Reproducible California Housing regression pipeline with deterministic evaluation, validated model artifacts, FastAPI, Streamlit and Docker.

Recommended topics:

`machine-learning`, `python`, `scikit-learn`, `fastapi`, `streamlit`, `regression`, `docker`, `model-serving`

Use a Homepage URL only after a current hosted deployment is freshly verified.

## 7. Interview explanation

Explain the project in this order:

1. why a simple baseline is useful;
2. why selection happens on validation rather than final test;
3. why Random Forest won the fixed comparison;
4. RMSE vs MAE vs R²;
5. deterministic seeds and same-environment reproducibility;
6. cleaning row accounting and why thresholds are demonstration filters rather than proven optimal outlier rules;
7. canonical feature order and the difference between training ranges and API/UI bounds;
8. joblib trusted-code boundary and exact scikit-learn compatibility check;
9. Pydantic validation and safe API failures;
10. feature importance versus causality;
11. historical-data/generalisation limits;
12. what production monitoring/governance would require.

## 8. Handing the project to another AI

Provide `AI_HANDOFF.md` and require the assistant to re-check current `main`, open pull requests, CI and deployment evidence before making any current claim. Do not resume stale project-order tasks from older handoffs.
