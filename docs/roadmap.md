# Roadmap

## Completed portfolio foundation

- deterministic data split and model seeds;
- Linear Regression baseline and Random Forest comparison;
- versioned model bundle and canonical feature schema;
- validated FastAPI and Streamlit prediction paths;
- offline tests, Python matrix CI, Ruff and dependency auditing;
- non-root Docker API image;
- security, model-card and portfolio documentation.

## Next engineering improvements

1. Add a dedicated integration workflow that downloads the real dataset, trains the model and publishes metrics as a CI artifact without committing generated binaries.
2. Add cross-validation and confidence intervals for model comparison.
3. Add data-schema validation with explicit missing-column and dtype reports.
4. Add signed model manifests or a governed model registry for production artifacts.
5. Add drift and out-of-distribution diagnostics for submitted features.
6. Add latency and prediction-distribution monitoring in a private deployment.
7. Add structured API versioning and OpenAPI examples.
8. Evaluate more transparent alternatives such as HistGradientBoosting and monotonic models.
9. Add fairness and geographic error analysis with appropriate legal and ethical review.

## Commercial boundary

A paid service should be implemented in a separate private repository with current licensed data, identity, rate limiting, tenant isolation, managed secrets, signed artifacts, monitoring, retention controls, incident response, legal review and independent security/model-risk testing.
