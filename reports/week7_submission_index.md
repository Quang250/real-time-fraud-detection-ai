# Week 7 Submission Index

## Core Implementation

- [x] `api/main.py` - FastAPI app with `/health`, `/model-info`, `/input-schema`, `/audit-summary`, `/predict`, `/predict-explain`, and `/predict-batch`
- [x] `api/schemas.py` - Pydantic request and response schemas
- [x] `api/audit_logger.py` - JSONL audit logging for successful predictions

## Test Scripts

- [x] `tests/test_api.py` - official pytest suite
- [x] `tests/manual_predict_with_sample_payload.py` - manual single-prediction success test
- [x] `tests/manual_predict_batch_with_sample.py` - manual batch-prediction success test
- [x] `tests/manual_bad_request_test.py` - missing-feature validation test
- [x] `tests/api_latency_test.py` - API latency benchmark
- [x] `tests/api_stream_simulation.py` - mixed stream simulation through the API
- [x] `tests/test_api.py` - includes `/predict-explain` and `/audit-summary` endpoint coverage

## Evidence Files

- [x] `reports/week7_predict_success_response.json`
- [x] `reports/week7_predict_batch_success_response.json`
- [x] `reports/week7_api_test_results.csv`
- [x] `reports/week7_api_latency_summary.csv`
- [x] `reports/week8_threshold_simulation.csv`
- [x] `reports/week8_cost_based_threshold_simulation.csv`
- [x] `reports/week7_api_stream_simulation.csv`
- [x] `reports/week8_predict_explain_response.json`
- [x] `reports/week8_audit_summary_response.json`
- [x] `reports/api_audit_log.jsonl`
- [x] `reports/week7_api_notes.md`
- [x] `reports/week7_api_summary.md`

## Runtime Logs

- [x] `reports/week7_uvicorn_stdout.log`
- [x] `reports/week7_uvicorn_stderr.log`

## Final Status

The Week 7 FastAPI service is working as a prototype inference layer for the
fraud detection model. It accepts model-ready features, validates inputs,
returns predictions, records audit logs, and is covered by pytest plus manual
verification scripts.
