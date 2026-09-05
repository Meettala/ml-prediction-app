# Roadmap

## Completed portfolio foundation

- deterministic train/validation/final-test separation with final test excluded from model selection;
- Linear Regression baseline and fixed Random Forest comparison on validation RMSE;
- auditable cleaning row counts and generated environment/split/metric evidence;
- real California Housing CI training, committed-metrics comparison and same-environment reproducibility check;
- versioned selected-model bundle, canonical feature order and exact scikit-learn compatibility guard;
- validated FastAPI and Streamlit prediction paths;
- offline tests, Python 3.10–3.12 matrix CI, `pip check`, Ruff and dependency auditing;
- non-root Docker API build/health/OpenAPI/prediction smoke;
- security, model-card and portfolio documentation.

## Possible future engineering work

These are not required to support the current portfolio claims:

1. Add repeated cross-validation or confidence intervals when a broader model-comparison study is actually needed.
2. Add richer data-drift/out-of-distribution diagnostics for submitted features.
3. Add signed model manifests or a governed registry for production artifacts.
4. Add latency and prediction-distribution monitoring in a private deployment.
5. Add structured API versioning and OpenAPI examples.
6. Evaluate additional transparent candidates only if a real product/research objective justifies them.
7. Add fairness and geographic error analysis with appropriate legal and ethical review.
8. Replace the historical dataset with current licensed data only in a governed use case that needs current-market estimation.

## Commercial boundary

A paid service should be implemented with current licensed data, identity, rate limiting, tenant isolation where relevant, managed secrets, governed/signed artifacts, monitoring, privacy/retention controls, incident response, legal review and independent security/model-risk testing.
