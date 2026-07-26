# ML Prediction App — California Housing

A reproducible regression application that moves beyond a notebook: public aggregate data, transparent cleaning, deterministic train/test evaluation, a simple baseline, a stronger model, versioned artifacts, FastAPI, Streamlit, Docker and explicit model limitations.

![ML Prediction App architecture](docs/assets/architecture.svg)

## Why this project exists

Many machine-learning portfolio projects stop at model training or show only the best metric. This project demonstrates the engineering around a model:

- a visible Linear Regression baseline and Random Forest comparison;
- held-out RMSE, MAE and R²;
- deterministic seeds and train/test separation;
- a canonical feature schema shared by training and serving;
- a versioned, validated model bundle;
- strict API and UI input handling;
- offline tests and dependency auditing;
- transparent model-risk documentation.

The prediction is an educational block-group estimate from 1990 US Census data. It is **not** a current property valuation, appraisal, investment recommendation or financial advice.

## Engineering highlights

- Public California Housing dataset from scikit-learn.
- Transparent duplicate removal and documented extreme-row filtering.
- Fixed train/test split and Random Forest random state.
- Linear Regression baseline with training-fitted standardisation.
- Random Forest used for the API and Streamlit estimate.
- Held-out RMSE, MAE and R² exported from the real pipeline.
- Canonical feature order in `src/mlapp/artifacts.py`.
- Versioned model metadata and schema validation.
- Rejection of extra, non-finite and out-of-range API values.
- Finite, non-negative prediction validation.
- Safe errors for missing, corrupt or incompatible artifacts.
- Python 3.10–3.12 tests, Ruff and `pip-audit` in GitHub Actions.
- Non-root Docker API image with a health check.

## Architecture

```text
Public aggregate dataset
        |
        v
Transparent cleaning
        |
        v
Deterministic train/test split
        |
   +----+----+
   |         |
Linear    Random Forest
baseline      |
   |          v
   +--> held-out metrics
              |
              v
     versioned model bundle
              |
        +-----+-----+
        |           |
     FastAPI     Streamlit
        |           |
        +-----v-----+
       canonical features
       finite estimate
       visible disclaimer
```

See [`docs/architecture.md`](docs/architecture.md) and [`docs/model-card.md`](docs/model-card.md).

## Quick start

### Requirements

- Python 3.10–3.12
- Internet access for the first real dataset download

```bash
git clone https://github.com/Meettala/ml-prediction-app.git
cd ml-prediction-app
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.mlapp.pipeline
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The training command writes trusted local outputs to:

```text
models/random_forest.joblib
models/linear_baseline.joblib
exports/metrics.json
```

`joblib` files can execute code while loading. Only load artifacts created by this trusted local pipeline; never accept uploaded model files or user-controlled artifact paths.

## Run the interfaces

FastAPI:

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`.

Streamlit:

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501`.

## Example API request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 5.0,
    "HouseAge": 20,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
  }'
```

The response contains the estimate in dataset units and US dollars, the model name and a disclaimer. Exact values depend on the locally trained artifact.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check src api streamlit_app tests
pip-audit -r requirements.txt
```

The unit suite is offline and deterministic. It covers:

- data loading through an injected synthetic frame;
- cleaning and reproducible splitting;
- model-bundle round trips and schema mismatch;
- missing and legacy artifact handling;
- non-finite and invalid predictions;
- API feature order;
- extra, non-finite and out-of-range requests;
- safe errors for missing artifacts and invalid model output.

Run `python -m src.mlapp.pipeline` separately for the real dataset/model integration path.

## Docker API demo

```bash
docker build -t ml-prediction-app .
docker run --rm -p 8000:8000 ml-prediction-app
```

The build downloads the public dataset and generates trusted model/metrics artifacts inside the image. The runtime stage uses a non-root user and serves FastAPI on port 8000.

## Repository structure

```text
.
├── src/mlapp/               # data, training, artifacts and pipeline
├── api/                     # FastAPI service
├── streamlit_app/           # interactive portfolio UI
├── tests/                   # offline unit and boundary tests
├── docs/                    # architecture, model card, roadmap and portfolio guide
├── .github/workflows/       # CI
├── Dockerfile
├── AI_HANDOFF.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## Model limitations

- The data reflects 1990 conditions, not current prices.
- Records are block-group aggregates, not individual properties.
- Historical geographic and socioeconomic patterns can encode structural inequities.
- One held-out split does not prove performance across time, regions or populations.
- Feature importance is not causal explanation.
- Inputs within accepted ranges can still produce inaccurate estimates.
- The public demo has no authentication, monitoring, rate limiting or production model governance.

See [`docs/model-card.md`](docs/model-card.md) for the full model-risk statement.

## Portfolio materials

- [`docs/PORTFOLIO_PRESENTATION_GUIDE.md`](docs/PORTFOLIO_PRESENTATION_GUIDE.md)
- [`docs/assets/architecture.svg`](docs/assets/architecture.svg)
- [`docs/assets/social-preview.svg`](docs/assets/social-preview.svg)
- [`AI_HANDOFF.md`](AI_HANDOFF.md)

## Commercial boundary

This public repository is MIT licensed for portfolio and reference use. A client-facing or paid service should be built in a separate private governed repository with current licensed data, identity, rate limiting, signed artifacts, monitoring, drift evaluation, legal review and independent security/model-risk testing.

See [`docs/commercialisation-and-private-production.md`](docs/commercialisation-and-private-production.md).

## Licence

MIT. See [`LICENSE`](LICENSE).

## Author

Built by [Meet Tala](https://github.com/Meettala) to demonstrate applied machine learning, evaluation, FastAPI, Streamlit, model-artifact validation, Docker and production-minded documentation.
