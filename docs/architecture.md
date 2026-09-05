# Architecture

## Training and serving flows

```text
California Housing dataset
        |
        v
Schema check + auditable fixed cleaning
        |
        v
Create final test split first (20%, random_state=42)
        |
        +-------------------------------------> untouched final test
        |
        v
Split development rows into train / validation
(validation = 25% of development, random_state=43)
        |
   +----+----+
   |         |
Linear    Random Forest
baseline    candidate
   |         |
   +----v----+
validation RMSE comparison
        |
        v
selected model
        |
        v
refit on train + validation
        |
        v
one final RMSE / MAE / R² evaluation
        |
        v
versioned trusted selected-model bundle
        |
   +----+----+
   |         |
FastAPI    Streamlit
   |         |
   +----v----+
canonical feature order
finite/non-negative output check
historical/non-valuation disclaimer
```

## Key modules

- `src/mlapp/data.py` loads the public aggregate dataset, validates schema, records sequential cleaning counts and creates deterministic train/validation/final-test partitions.
- `src/mlapp/train.py` fits the fixed Linear Regression and Random Forest candidates, evaluates them on validation and applies the model-selection rule.
- `src/mlapp/artifacts.py` defines canonical feature order, the versioned selected-model bundle, exact scikit-learn compatibility check and output validation.
- `src/mlapp/pipeline.py` orchestrates real-data training, validation selection, selected-model refit, final test evaluation and generated metrics/artifact output.
- `api/main.py` validates requests and serves the trusted local selected-model bundle.
- `streamlit_app/app.py` separates validation comparison from final-test evidence and presents model inspection, input controls and limitations.
- `tools/verify_reproducibility.py` reruns the real pipeline and requires the metrics output to match byte-for-byte in the same validated environment.

## Evidence flow

`exports/metrics.json` distinguishes:

- raw/cleaned row accounting;
- split sizes and purposes;
- fixed model parameters;
- validation metrics used for selection;
- selected model and criterion;
- final held-out test metrics;
- exact training-library versions;
- observed cleaned feature ranges;
- API demo and Streamlit convenience ranges;
- selected-model feature importance and sample predictions;
- limitations.

CI preserves the committed metrics file, runs the real pipeline, compares the regenerated file with the committed evidence, then runs the pipeline again to verify deterministic reproduction.

## Trust boundaries

1. Request values are untrusted and validated by Pydantic.
2. Feature order is canonical and independent of JSON request order.
3. API demo ranges and Streamlit convenience ranges are serving controls, not the training-data schema.
4. Model outputs must be numeric, finite and non-negative.
5. Joblib/pickle artifacts are trusted local build outputs only. Exact artifact/runtime scikit-learn matching reduces compatibility ambiguity but does **not** make untrusted pickle files safe.
6. No model upload or user-controlled artifact path exists.
7. The public dataset contains historical aggregate block-group data, not individual property records.
8. Output is an illustrative historical median block-group estimate, not a current property valuation or financial recommendation.

## Reproducibility

- Final test random state: 42.
- Validation random state: 43.
- Random Forest random state: 42.
- Candidate models and parameters are fixed before validation comparison.
- Unit/integration tests use synthetic frames/dummy models and avoid hidden network training.
- Real California Housing training runs in a dedicated CI evidence job.
- Exact validated environment is recorded in generated metrics; arbitrary cross-version bit-for-bit reproducibility is not claimed.

## Production boundary

The public project is not a production MLOps platform. A commercial implementation would need current licensed data, provenance/freshness controls, governed retraining, signed/managed artifacts, authentication, rate limiting, monitoring, drift/performance evaluation, privacy/retention controls, incident response, legal review and independent security/model-risk assessment.
