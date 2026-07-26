# AI Handoff — ML Prediction App

> Paste this file into another AI assistant to continue the project without restarting it. Verify the live repository, active pull requests, branch head, dependencies, model artifacts and CI before editing.

## Continuation instruction

You are continuing `Meettala/ml-prediction-app`, a public MIT-licensed portfolio project owned by Meet Tala.

Preserve reproducible data preparation, train/test separation, schema-validated predictions, transparent model limitations and the public aggregate-data boundary. Read the live implementation, tests, `README.md`, `SECURITY.md`, `docs/architecture.md`, `docs/model-card.md`, `docs/roadmap.md` and `docs/PORTFOLIO_PRESENTATION_GUIDE.md` before changing behaviour. Add tests for material changes and update this handoff after code, security, dependency, model, deployment, documentation or presentation work.

Never commit private datasets, secrets, customer records, production endpoints or confidential model artifacts.

## Repository state

- Repository: `Meettala/ml-prediction-app`
- Default branch: `main`
- Working branch: `agent/professional-repository-foundation`
- Active pull request: Draft PR #1, `Professionalize ML prediction app`
- Starting commit: `c1c900bb3f84eb4c9b5000bcab216dd894cf396b`
- Visibility: public
- Stack: Python, pandas, scikit-learn, FastAPI, Streamlit, Pydantic, pytest, Ruff and Docker
- Dataset: scikit-learn California Housing aggregate block-group dataset
- Licence: MIT on the working branch
- Last updated: 26 July 2026

## Product purpose

The project trains and evaluates housing-value regression models, saves a selected model, exposes estimates through FastAPI and Streamlit, and exports metrics for portfolio use.

The output is an educational estimate from 1990 census block-group data. It is not a property valuation, appraisal, financial recommendation or guarantee.

## Core trust model

1. Public aggregate data only in the public repository.
2. Transparent cleaning before a deterministic train/test split.
3. The Linear Regression baseline and Random Forest are evaluated on held-out data.
4. Feature order is defined centrally in `src/mlapp/artifacts.py`.
5. The Random Forest is saved inside a versioned bundle with model name, feature schema and scikit-learn metadata.
6. Joblib validation detects stale or incompatible metadata but does not make untrusted pickle files safe. Only trusted local pipeline artifacts may be loaded.
7. FastAPI rejects extra, non-finite and out-of-range values.
8. API and Streamlit predictions must be numeric, finite and non-negative.
9. Missing, corrupt or incompatible artifacts produce safe user-facing errors.
10. Documentation and interfaces must show the historical-data and non-valuation limitation.

## Implemented on the professionalisation branch

### Model and reliability

- Added `src/mlapp/artifacts.py` with the canonical feature schema, versioned bundle and prediction validation.
- Updated the training pipeline to write the validated Random Forest bundle.
- Added configurable model and metrics output paths for isolated tests.
- Preserved the visible Linear Regression baseline and deterministic Random Forest.
- Hardened empty sample-prediction handling.

### API and Streamlit

- FastAPI now forbids extra fields and rejects non-finite/out-of-range inputs.
- API construction follows the canonical training feature order.
- Missing artifacts return a generic 503; invalid model output returns a generic 500.
- Streamlit loads the same validated bundle and safely handles invalid metrics/model output.
- The UI labels the result as an educational historical block-group estimate.

### Tests and quality

- Added offline model-bundle and prediction-validation tests.
- Replaced network/training work hidden inside API test imports with dummy-model boundary tests.
- Reworked data tests to use injected synthetic frames.
- Added Python 3.10–3.12 GitHub Actions, Ruff and `pip-audit`.
- Existing baseline tests passed on Python 3.10–3.12 before hardening; the integrated head still requires final verification.

### Deployment, governance and portfolio

- Added a non-root multi-stage Docker API image with a health check.
- Added MIT `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md` and a pull-request template.
- Added architecture, model card, roadmap and commercial/private-production documentation.
- Added architecture and social-preview SVG assets.
- Added a beginner-friendly portfolio presentation guide.
- Reworked the README for recruiters and technical reviewers.

## Decisions to preserve

- Use public aggregate or explicitly permitted datasets only.
- Keep train/test separation and deterministic evaluation visible.
- Keep a simple baseline beside the stronger model.
- Predictions are estimates, not valuations, financial advice or guarantees.
- Do not add model-upload endpoints or user-controlled joblib paths.
- Documentation claims require code, tests, CI or measured evidence.
- Future commercial work belongs in a separate private governed repository.
- Do not claim production readiness, fairness across all populations or universal predictive accuracy.

## Known limitations

- The data is from 1990 and cannot represent current prices.
- Records are census block-group aggregates, not individual properties.
- The current evaluation uses one deterministic held-out split rather than a full cross-validation study.
- Historical geographic and socioeconomic patterns can encode structural inequities.
- Random Forest importance is not causal explanation.
- The public demo has no authentication, rate limiting, monitoring or governed model registry.
- The Docker build downloads the public dataset and requires network access.
- Real screenshot/video capture and GitHub social-preview upload remain manual presentation tasks.

## Immediate next work

1. Confirm tests, Ruff and `pip-audit` on the exact integrated branch head.
2. Fix every finding without weakening the gates.
3. Remove temporary Ruff diagnostic-artifact plumbing.
4. Review Docker build assumptions and repository documentation for consistency.
5. Update the PR description with the exact completed scope and verified CI result.
6. Mark PR #1 ready and squash-merge only after the exact final head is green and mergeable.
7. After merge, capture a real Streamlit screenshot/demo and upload the rendered social-preview PNG manually.

## Rules for another AI

Before editing, inspect the live branch, PR, CI and implementation rather than trusting this summary alone. Keep changes scoped, add positive and negative tests, never expose private data or secrets, and update this file and the PR after material work.
