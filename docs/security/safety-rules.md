# Safety rules — ML Prediction App

1. This project is an illustrative portfolio piece, not financial or
   investment advice. Every surface that shows a prediction (API response,
   Streamlit app, portfolio demo) says so explicitly.
2. Both the baseline (Linear Regression) and the primary model (Random
   Forest) evaluation metrics are always shown together — never just the
   best-looking number. See `exports/metrics.json`.
3. The dataset is public, aggregate, block-group-level 1990 census data —
   no individual, identifiable person's data is used or could be
   reconstructed from it.
4. Model limitations (dataset age, geographic granularity, illustrative
   scope) are surfaced in-product, not buried in a README.
5. Input validation (`api/main.py`) rejects out-of-range values rather than
   silently producing a nonsensical prediction.
