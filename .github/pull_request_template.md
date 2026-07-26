## Summary

Describe the change and the model/data/API risk it addresses.

## Validation

- [ ] Python 3.10–3.12 tests pass
- [ ] Ruff passes
- [ ] Runtime dependency audit passes or any exception is narrowly documented
- [ ] New behaviour has positive and negative tests
- [ ] Documentation and model card are updated where needed

## ML and safety review

- [ ] Train/test separation is preserved
- [ ] Randomness and reproducibility effects are documented
- [ ] Feature order and artifact compatibility are preserved or migrated
- [ ] No private or restricted data is introduced
- [ ] Predictions are not presented as valuations, financial advice or guarantees
- [ ] No untrusted pickle/joblib upload or user-controlled artifact path is added

## Evidence

List tests, metrics, screenshots or other evidence supporting the change. Do not include secrets or private data.
