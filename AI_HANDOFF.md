# AI Handoff — ML Prediction App

> Use this file to continue `Meettala/ml-prediction-app` without restarting the project. Always verify the live repository, current branch, CI and deployments before editing.

## Current repository state

- Repository: `Meettala/ml-prediction-app`
- Default branch: `main`
- Visibility: public
- Licence: MIT
- Stack: Python, pandas, scikit-learn, FastAPI, Streamlit, Pydantic, pytest, Ruff and Docker
- Dataset: scikit-learn California Housing aggregate block-group dataset
- Last updated: 1 August 2026

## Verified live deployments

- Streamlit UI: `https://meet-tala-ml-prediction-app-px2pch6zms5lrdhxyxurht.streamlit.app`
- FastAPI service: `https://ml-prediction-app-xfsd.onrender.com`
- Swagger/OpenAPI documentation: `https://ml-prediction-app-xfsd.onrender.com/docs`
- Health endpoint: `https://ml-prediction-app-xfsd.onrender.com/health`

The Streamlit UI was manually confirmed to load successfully after the source-import fix. The Render API, health endpoint, Swagger documentation, valid prediction request and invalid-input validation were also tested. Render uses a free instance and may take additional time to wake after inactivity.

## Product purpose

The project trains and evaluates housing-value regression models, saves a selected model, exposes estimates through FastAPI and Streamlit, and exports measured metrics for portfolio use.

The output is an educational estimate from 1990 census block-group data. It is not a property valuation, appraisal, financial recommendation or guarantee.

## Model and data results

Completed real pipeline run:

- Cleaned rows: 20,597
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Linear Regression: RMSE 0.6541, MAE 0.4867, R² 0.6786
- Random Forest: RMSE 0.4762, MAE 0.3193, R² 0.8297
- Selected model: Random Forest
- Generated artifacts:
  - `models/random_forest.joblib` — approximately 90 MB
  - `models/linear_baseline.joblib`
  - `exports/metrics.json`

Do not commit unnecessary downloaded datasets or cache files. The large Random Forest artifact should only be committed if repository policy explicitly requires it.

## Core trust model

1. Use public aggregate or explicitly permitted data only.
2. Preserve transparent cleaning and deterministic train/test separation.
3. Keep the Linear Regression baseline visible beside the stronger model.
4. Define feature order centrally in `src/mlapp/artifacts.py`.
5. Save the Random Forest in a versioned bundle with model name, feature schema and scikit-learn metadata.
6. Treat joblib/pickle artifacts as trusted-code files. Never accept uploaded model files or user-controlled artifact paths.
7. Reject extra, non-finite and out-of-range API values.
8. Require predictions to be numeric, finite and non-negative.
9. Return safe generic errors for missing, corrupt or incompatible artifacts.
10. Display the historical-data and non-valuation limitation in documentation, API output and UI.

## Deployment history

### FastAPI and Docker

- Docker build initially failed because builder dependencies installed under `/install` were not on the pipeline import path.
- Fixed with `PYTHONPATH=/install/lib/python3.12/site-packages:/app python -m src.mlapp.pipeline`.
- Docker build and runtime were validated.
- `/health`, `/docs`, `/openapi.json`, valid `/predict`, invalid request handling and artifact presence passed.
- FastAPI deployed to Render at the verified URL above.
- Opening the base Render URL returns `{"detail":"Not Found"}` because the API intentionally has no root homepage route.

### Streamlit

- Streamlit Community Cloud initially failed with `ModuleNotFoundError: No module named 'src'`.
- PR #4 added the repository root to `sys.path` before importing `src.mlapp`, with narrow Ruff `E402` exemptions.
- PR #4 passed CI and merged as commit `9fd5bb8b5f6c9cd97865a29540c2c4771dc89d1e`.
- The app was renamed from a misleading `jobpilot-ai` URL to the verified ML Prediction App URL above.
- User confirmed the deployed California Housing interface loads successfully.

## Completed validation

- Python 3.10, 3.11 and 3.12 tests passing
- Ruff passing
- Dependency auditing passing at the last checked CI state
- Docker build and startup passing
- FastAPI health and docs passing
- Valid API prediction returning HTTP 200
- Invalid API cases returning HTTP 422
- Streamlit page loading and displaying model comparison, feature importance, sliders, prediction and limitations

Verified sample request:

```json
{
  "MedInc": 5.0,
  "HouseAge": 20,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1500,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```

A completed API test returned a Random Forest estimate of approximately `$232,934`, with the historical-data disclaimer. Exact values depend on the deployed artifact.

## Known limitations

- The data is from 1990 and cannot represent current prices.
- Records are census block-group aggregates, not individual properties.
- The evaluation uses one deterministic held-out split rather than a full temporal or cross-validation study.
- Historical geographic and socioeconomic patterns can encode structural inequities.
- Random Forest feature importance is not causal explanation.
- The public deployments have no authentication, rate limiting, monitoring or governed model registry.
- The Render free instance may sleep after inactivity.
- Real screenshots, demo video and social-preview upload remain presentation tasks.

## Next work

1. Merge the documentation PR that adds verified live links to `README.md` and this handoff.
2. Add the Streamlit live link and GitHub repository to the portfolio project card.
3. Add a concise project entry to the CV and LinkedIn Projects section.
4. Capture screenshots of model comparison, feature importance, prediction output, limitations, API docs, health response and CI.
5. Keep all future claims tied to code, tests, measured metrics or verified deployments.

## Suggested CV entry

**ML Prediction App — California Housing**  
Built and deployed an end-to-end regression application using Python, scikit-learn, FastAPI, Streamlit and Docker. Compared Linear Regression with Random Forest on a held-out test set, achieving an R² of 0.83 with the selected model. Added schema-validated API inputs, versioned model artifacts, CI across Python 3.10–3.12 and public deployments on Streamlit Community Cloud and Render.

## Suggested LinkedIn project description

Developed a production-minded machine-learning portfolio application using the California Housing dataset. The project includes reproducible training, baseline comparison, validated model artifacts, FastAPI, Streamlit, Docker, automated testing and public deployment. Random Forest achieved a held-out R² of approximately 0.83. The interface clearly states that results are historical block-group estimates rather than current property valuations or financial advice.

## Rules for another AI

Inspect the live repository and CI before editing. Preserve the data, evaluation, artifact-validation and disclaimer boundaries. Add positive and negative tests for material behaviour changes. Never expose secrets, private data or untrusted pickle files. Update this handoff after code, deployment, dependency, documentation or presentation work.
