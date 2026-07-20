# MVP scope — ML Prediction App

## In scope
- End-to-end pipeline: load, clean, split, train two models, evaluate.
- Honest baseline comparison (Linear Regression vs. Random Forest).
- FastAPI serving endpoint with input validation.
- Streamlit interactive demo using the real trained model.
- Static, client-side interactive demo on the portfolio site using the
  interpretable linear model's coefficients (clearly labeled as the
  simplified model — the reported accuracy is the Random Forest's).

## Explicitly out of scope
- Hyperparameter tuning / model selection beyond two reasonable baselines
  (the point of this project is the full clean pipeline, not squeezing
  out maximum accuracy).
- Real-time/current housing data (the dataset is a fixed 1990 census
  snapshot).
- Any actual financial or investment guidance.
