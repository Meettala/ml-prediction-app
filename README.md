# ML Prediction App — California Housing Price Predictor

A clean, end-to-end ML pipeline — from raw data to a working demo anyone
can try, not just a notebook. Predicts median house value for a census
block group from 1990 US Census data.

## Why this dataset

scikit-learn's built-in California Housing dataset: public, well-known,
aggregate (no personal data), and genuinely predictive — good for showing
a full pipeline rather than a toy task. See `docs/security/privacy-by-design.md`.

## Pipeline

1. `src/mlapp/data.py` — load, clean (drop duplicates, clip known outlier
   artifacts), split.
2. `src/mlapp/train.py` — train a Linear Regression baseline and a Random
   Forest, evaluate both with RMSE/MAE/R².
3. `src/mlapp/pipeline.py` — runs it all, saves model artifacts to
   `models/`, exports `exports/metrics.json` (consumed by the Streamlit
   app, the API, and the portfolio site's static demo).
4. `api/main.py` — FastAPI service (`/predict`, `/health`) with input
   validation.
5. `streamlit_app/app.py` — interactive local demo using the real trained
   model, with sliders, feature importance, and limitations shown inline.

## Run it

```bash
pip install -r requirements.txt
python -m src.mlapp.pipeline          # train + export metrics
uvicorn api.main:app --reload         # API at http://localhost:8000/docs
streamlit run streamlit_app/app.py    # interactive dashboard
```

## Tests

```bash
python -m pytest tests/ -q
```

## Docs

- [`docs/security/safety-rules.md`](docs/security/safety-rules.md)
- [`docs/security/privacy-by-design.md`](docs/security/privacy-by-design.md)
- [`docs/testing/test-notes.md`](docs/testing/test-notes.md)
- [`docs/product/mvp-scope.md`](docs/product/mvp-scope.md)
