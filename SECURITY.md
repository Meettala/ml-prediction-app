# Security Policy

## Supported scope

This repository is a public portfolio and reference implementation. Security fixes are applied to the current `main` branch.

## Report a vulnerability

Do not open a public issue containing exploit details, private data, credentials or malicious model artifacts. Use GitHub's private vulnerability reporting feature when available, or contact the repository owner privately.

Include:

- affected file or endpoint;
- reproduction steps using non-sensitive data;
- expected impact;
- suggested mitigation, if known.

## Trust boundaries

- API and Streamlit feature values are untrusted and must pass schema/range validation.
- `joblib`/pickle files can execute code while loading. Only artifacts created by this repository's trusted training pipeline may be loaded. Never accept uploaded model artifacts or user-controlled artifact paths.
- The versioned artifact records canonical features, model name and scikit-learn version. Loading requires an exact runtime scikit-learn version match and otherwise asks for retraining. This is a compatibility guard, **not** a sandbox or security guarantee for untrusted pickle/joblib files.
- Missing, corrupt, incompatible or invalid-output artifacts are surfaced through generic API errors rather than raw deserialization exceptions or filesystem paths.
- The California Housing dataset is public aggregate census data. Do not replace it with private or restricted data in the public repository.
- Generated metrics and estimates are based on 1990 block-group data, not current property valuations or financial advice.
- Secrets, private datasets and production infrastructure details must not be committed.

## Known limitations

This demonstration does not include authentication, rate limiting, tenant isolation, encrypted persistence, production drift/performance monitoring, independent penetration testing or a complete model-risk governance programme. A commercial service requires a separate governed implementation with current licensed data and operational controls.
