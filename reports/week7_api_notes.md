# Week 7 - FastAPI Fraud Detection Service

## Objective

The objective of this stage is to expose the trained fraud detection model
through a FastAPI service. This allows the model to receive transaction data
through HTTP requests and return fraud prediction results in JSON format.

## API Endpoints

- `GET /`: root endpoint
- `GET /health`: check service and model artifact status
- `GET /model-info`: return model metadata
- `GET /input-schema`: return required feature list
- `GET /audit-summary`: summarize audit log activity and latency
- `POST /predict`: predict fraud risk for one transaction
- `POST /predict-explain`: predict fraud risk and return top SHAP feature contributions
- `POST /predict-batch`: predict fraud risk for multiple transactions

## Input Format

The API expects model-ready features with the same 47 features used during
model training.

## Output Format

The API returns:

- transaction ID
- fraud probability
- binary prediction
- prediction label
- decision threshold
- latency in milliseconds
- model version
- top feature explanations for `/predict-explain`

## Audit Logging

Each prediction request is recorded in `reports/api_audit_log.jsonl` to support
traceability and later review.

`GET /audit-summary` reads this log and returns total requests, fraud
predictions, non-fraud predictions, average latency, P95 latency, and the
latest transaction ID.

## Current Limitation

The current API is a prototype. It expects model-ready features rather than raw
transaction data. In a real-world deployment, a feature engineering service or
feature store would generate these features before calling the model API.
