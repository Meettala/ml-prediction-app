# Architecture

## Request and training flows

```text
California Housing dataset
        |
        v
Schema check + transparent cleaning
        |
        v
Deterministic train/test split
        |
        +--> Linear Regression baseline
        |
        +--> Random Forest candidate
                 |
                 v
        Held-out RMSE / MAE / R²
                 |
                 v
       Versioned trusted model bundle
                 |
          +------+------+
          |             |
       FastAPI       Streamlit
          |             |
          +------v------+
          validated feature order
          finite prediction check
          disclaimer + limitations
```

## Key modules

- `src/mlapp/data.py` loads, cleans and splits the public aggregate dataset.
- `src/mlapp/train.py` trains the baseline and Random Forest and calculates held-out metrics.
- `src/mlapp/artifacts.py` defines the canonical feature schema, versioned bundle and prediction validation.
- `src/mlapp/pipeline.py` orchestrates training and writes model/metrics artifacts.
- `api/main.py` validates requests and serves the trusted local model.
- `streamlit_app/app.py` presents metrics, feature importance, input controls and limitations.

## Trust boundaries

1. Request values are untrusted and validated by Pydantic.
2. Feature order is fixed centrally rather than inferred from request order.
3. Model outputs are converted to finite, non-negative values before use.
4. Joblib artifacts are trusted local build outputs only. Metadata validation catches corruption or schema drift but does not make untrusted pickle files safe.
5. The public dataset contains aggregate block-group data, not personal records.
6. Predictions are historical-model estimates, not property valuations or financial advice.

## Reproducibility

- The train/test split uses a fixed random state.
- The Random Forest uses a fixed random state.
- Unit tests use synthetic frames and dummy models, avoiding hidden network/training work.
- The real pipeline remains available as an explicit integration command.

## Production boundary

The public project is not a complete production service. A commercial implementation needs governed data refreshes, model registry and signatures, authentication, abuse controls, monitoring, drift evaluation, incident response, legal review and independent security/model-risk assessment.
