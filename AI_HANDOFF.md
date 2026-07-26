# AI Handoff — ML Prediction App

> Paste this file into another AI assistant to continue the project without restarting it. Verify the live repository, active pull requests, branch head, dependencies, model artifacts and CI before editing.

## Continuation instruction

You are continuing `Meettala/ml-prediction-app`, a public portfolio project owned by Meet Tala.

Preserve reproducible data preparation, train/test separation, schema-validated predictions, transparent model limitations and the public aggregate-data boundary. Read the live implementation, tests, README and security documents before changing behaviour. Add tests for material changes and update this handoff after code, security, dependency, model, deployment, documentation or presentation work.

Never commit private datasets, secrets, customer records, production endpoints or confidential model artifacts.

## Repository state

- Default branch: `main`
- Working branch: `agent/professional-repository-foundation`
- Starting commit: `c1c900bb3f84eb4c9b5000bcab216dd894cf396b`
- Existing pull requests before this work: none
- Visibility: public
- Starting stack: Python, pandas, scikit-learn, FastAPI, Streamlit, pytest
- Dataset described by README: scikit-learn California Housing aggregate dataset
- Last updated: 26 July 2026

## Product purpose

The project trains and evaluates housing-value regression models, saves a selected model, exposes predictions through FastAPI and Streamlit, and exports metrics for portfolio use.

## Audit and completion plan

1. Inspect data loading, preprocessing, splitting, training, evaluation and artifact loading.
2. Verify leakage prevention, deterministic seeds and reproducible model selection.
3. Validate metric calculations, units and model-card claims.
4. Harden model serialization and version compatibility.
5. Validate API and Streamlit input schemas, ranges, non-finite values and error handling.
6. Add tests for malformed input, missing/corrupt artifacts, feature order, prediction bounds and repeatability.
7. Establish Python 3.10–3.12 CI, Ruff and dependency auditing.
8. Add packaging, Docker support, health checks and repository governance.
9. Rework README and add architecture, roadmap, model card, presentation assets and portfolio guide.
10. Keep the PR draft until the exact final head is green and the security/recruiter review is complete.

## Decisions to preserve

- Use public aggregate or explicitly permitted datasets only.
- Keep train/test separation and deterministic evaluation visible.
- Predictions are estimates, not valuations, financial advice or guarantees.
- Input and model artifacts are untrusted until validated.
- Do not claim production readiness, fairness across all populations or universal predictive accuracy.
- Future commercial work belongs in a separate private governed repository.
