# Week 7 API Summary

## Endpoints

The FastAPI service exposes the following endpoints:

- `GET /`
- `GET /health`
- `GET /model-info`
- `GET /input-schema`
- `GET /audit-summary`
- `POST /predict`
- `POST /predict-explain`
- `POST /predict-batch`

## Validation

- `POST /predict` returns `200` when the request includes all 47 model-ready features.
- `POST /predict-explain` returns `200` with prediction output and top SHAP feature contributions.
- `GET /audit-summary` returns monitoring summary from the JSONL audit log.
- The API returns `422` for invalid input, including missing required features.

## Logging

- Every successful prediction is written to `reports/api_audit_log.jsonl`.
- The audit log records timestamp, transaction ID, model version, fraud probability,
  prediction, prediction label, threshold, and latency.

## Tests and Evidence

- Single prediction and batch prediction were both verified with `200` responses.
- Latency testing was completed and summarized in `reports/week7_api_latency_summary.csv`.
- Mixed stream simulation was completed and saved in `reports/week7_api_stream_simulation.csv`.
- Threshold and cost-based simulations were added for risk-management review.
- Audit summary response was saved in `reports/week8_audit_summary_response.json`.
- Pytest was run successfully against `tests/test_api.py`.

## Result

The API is working as a prototype fraud detection inference service with
validation, audit logging, latency measurement, and automated testing in place.
