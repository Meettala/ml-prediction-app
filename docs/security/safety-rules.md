# Safety rules — ML Prediction App

1. This project is an illustrative historical portfolio model, not a property valuation, appraisal, financial recommendation or investment advice. Prediction surfaces say so explicitly.
2. Evaluation roles must not be blurred: candidate metrics are reported on validation for selection, while only the already-selected refit model is reported on the untouched final test. See `exports/metrics.json`.
3. Never describe R² as percent prediction accuracy, percent confidence or percent of properties predicted correctly.
4. The dataset is public, aggregate, block-group-level 1990 census data; it is not current market data or individual property data.
5. Cleaning thresholds are transparent fixed demonstration filters, not statistically proven optimal outlier rules.
6. Random Forest impurity importance is model inspection, not causal evidence.
7. Joblib/pickle artifacts are trusted local code artifacts only. Never add an untrusted model upload or user-controlled artifact path.
8. Input validation rejects missing, extra, non-finite and out-of-range values rather than silently producing an estimate from malformed input.
9. Safe serving errors must not reveal local filesystem paths or raw deserialization exceptions.
10. Model limitations (dataset age, geographic granularity, one deterministic final split and illustrative scope) are surfaced in-product, not buried only in documentation.
