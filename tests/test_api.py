"""Pytest test suite for the FastAPI fraud detection API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "reports" / "sample_transaction_payload.json"


@pytest.fixture
def client():
    """Create a TestClient that runs the FastAPI lifespan startup."""
    with TestClient(app) as test_client:
        yield test_client


def load_sample_features():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["threshold_loaded"] is True
    assert data["feature_count"] == 47


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["threshold"] == 0.73
    assert data["feature_count"] == 47
    assert data["input_type"] == "model_ready_features"


def test_single_prediction(client):
    features = load_sample_features()

    payload = {
        "transaction_id": "pytest_tx_001",
        "features": features,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == "pytest_tx_001"
    assert "fraud_probability" in data
    assert "prediction" in data
    assert "prediction_label" in data
    assert data["threshold"] == 0.73
    assert data["model_version"] == "xgboost_v1_threshold_0.73"


def test_predict_explain(client):
    features = load_sample_features()

    payload = {
        "transaction_id": "pytest_explain_tx_001",
        "features": features,
    }

    response = client.post("/predict-explain?top_n=5", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == "pytest_explain_tx_001"
    assert "fraud_probability" in data
    assert "prediction_label" in data
    assert data["threshold"] == 0.73
    assert len(data["top_explanations"]) == 5

    explanation = data["top_explanations"][0]
    assert "feature" in explanation
    assert "feature_value" in explanation
    assert "shap_value" in explanation
    assert explanation["direction"] in {
        "pushes toward Fraud",
        "pushes toward Non-fraud",
        "neutral",
    }


def test_batch_prediction(client):
    features = load_sample_features()

    payload = {
        "transactions": [
            {
                "transaction_id": "pytest_batch_tx_001",
                "features": features,
            },
            {
                "transaction_id": "pytest_batch_tx_002",
                "features": features,
            },
        ]
    }

    response = client.post("/predict-batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["total_transactions"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["transaction_id"] == "pytest_batch_tx_001"
    assert data["results"][0]["threshold"] == 0.73


def test_audit_summary(client):
    response = client.get("/audit-summary")

    assert response.status_code == 200

    data = response.json()

    assert "total_requests" in data
    assert "fraud_predictions" in data
    assert "non_fraud_predictions" in data
    assert "mean_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "latest_transaction_id" in data
    assert data["total_requests"] >= 0


def test_missing_feature_error(client):
    features = load_sample_features()
    features.pop("Time", None)

    payload = {
        "transaction_id": "pytest_bad_tx",
        "features": features,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
