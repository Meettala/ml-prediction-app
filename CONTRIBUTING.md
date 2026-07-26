# Contributing

Thank you for improving this portfolio reference project.

## Development workflow

1. Create a focused branch.
2. Install `requirements-dev.txt`.
3. Add or update tests for behavioural changes.
4. Run `python -m pytest`.
5. Run `ruff check src api streamlit_app tests`.
6. Run `pip-audit -r requirements.txt`.
7. Open a pull request describing model, data, API and documentation effects.

## Required standards

- Preserve deterministic train/test separation and fixed random seeds unless the change explicitly evaluates an alternative.
- Do not fit preprocessing on test data.
- Keep the canonical feature order in `src/mlapp/artifacts.py` aligned with the dataset schema.
- Treat model artifacts as trusted local build outputs only; never add upload endpoints for pickle/joblib files.
- Keep tests offline and deterministic. Real dataset/model integration runs should be explicit rather than hidden inside unit-test imports.
- Do not claim that predictions are current valuations, financial advice, universally fair or production-ready.
- Use public aggregate or explicitly permitted data only.
- Never commit secrets, private data or confidential infrastructure details.

## Pull-request evidence

A pull request should state:

- what changed;
- which failure or risk it addresses;
- tests added or updated;
- CI status;
- model/metric compatibility effects;
- documentation changes.
