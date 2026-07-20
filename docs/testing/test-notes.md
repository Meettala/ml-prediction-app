# Testing notes — ML Prediction App

- `tests/test_data.py` — data loads with expected shape/columns; cleaning
  removes known outlier artifacts without dropping most of the data.
- `tests/test_train.py` — Random Forest clears a sanity R² floor (catches
  a badly broken pipeline); Linear baseline is confirmed to under- or
  match- perform the Random Forest, not silently beat it due to a bug.
- `tests/test_api.py` — health check, a valid prediction request, and a
  rejected out-of-range request (422), using FastAPI's TestClient.

Run with:
```bash
python -m pytest tests/ -q
```
