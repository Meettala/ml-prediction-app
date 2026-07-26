# Portfolio Presentation Guide

This guide covers the manual presentation work that requires a running application or GitHub settings.

## 1. Run the real training pipeline

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

The first run downloads the public California Housing dataset and writes trusted local artifacts to `models/` and `exports/`.

## 2. Run the demos

FastAPI:

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` and test `/health` and `/predict`.

Streamlit:

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501`.

Docker API:

```bash
docker build -t ml-prediction-app .
docker run --rm -p 8000:8000 ml-prediction-app
```

## 3. Capture portfolio media

Capture one clean Streamlit screenshot showing:

- project title and historical-data disclaimer;
- held-out metrics;
- feature-importance chart;
- input controls;
- one plausible estimate;
- visible limitations.

Record a 30–60 second demonstration:

1. explain the dataset and target;
2. show the baseline and Random Forest metrics;
3. change two or three inputs;
4. show the estimate;
5. finish with the limitations and trusted-artifact boundary.

Do not present the output as a current property valuation.

## 4. GitHub social preview

Convert `docs/assets/social-preview.svg` to a 1280×640 PNG using a browser, design tool or trusted SVG converter. In GitHub:

1. open the repository;
2. choose **Settings**;
3. find **Social preview**;
4. upload the PNG;
5. save.

## 5. Suggested repository metadata

Description:

> Reproducible California Housing regression pipeline with validated model artifacts, FastAPI, Streamlit, Docker and transparent limitations.

Suggested topics:

`machine-learning`, `scikit-learn`, `fastapi`, `streamlit`, `regression`, `model-card`, `docker`, `python`, `mlops`

## 6. CV wording

> Built a reproducible regression application using scikit-learn, FastAPI and Streamlit; implemented held-out evaluation, versioned model artifacts, strict feature validation, offline tests, Docker deployment and model-risk documentation.

## 7. Interview explanation

Explain the project in this order:

1. why a notebook alone was not enough;
2. how train/test separation and fixed seeds protect evaluation;
3. why a simple baseline is shown next to the stronger model;
4. how feature order and artifact metadata are validated;
5. why joblib files are trusted local outputs only;
6. how API/UI errors avoid exposing internals;
7. why the model cannot be used as a current valuation tool.

## 8. Handing the project to another AI

Paste the complete `AI_HANDOFF.md` into the other assistant and ask it to verify the live `main` branch, open pull requests and latest CI before changing anything.
