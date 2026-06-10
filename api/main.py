"""FastAPI app for real-time fraud detection inference."""

import json
import math
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.audit_logger import write_audit_log
from api.schemas import (
    AuditSummaryResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationItem,
    HealthResponse,
    ModelInfoResponse,
    PredictionExplainResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.inference_pipeline import FraudInferencePipeline


MODEL_VERSION = "xgboost_v1_threshold_0.73"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_LOG_PATH = PROJECT_ROOT / "reports" / "api_audit_log.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API starts."""
    app.state.pipeline = FraudInferencePipeline()
    yield


app = FastAPI(
    title="Real-Time Credit Card Fraud Detection API",
    description=(
        "A FastAPI prototype for near real-time credit card fraud detection "
        "using an XGBoost model and model-ready transaction features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# CORS configuration for future Streamlit dashboard.
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline() -> FraudInferencePipeline:
    """Safely retrieve the loaded inference pipeline."""
    pipeline = getattr(app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline is not loaded.",
        )

    return pipeline


def _load_audit_records() -> list[Dict[str, Any]]:
    """Read JSON Lines audit records and ignore malformed rows."""
    if not AUDIT_LOG_PATH.exists():
        return []

    records = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """Return a simple nearest-rank percentile for monitoring summaries."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = math.ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def _predict_one(payload: TransactionRequest) -> PredictionResponse:
    """Run prediction for one transaction request."""
    pipeline = _get_pipeline()
    transaction_id = payload.transaction_id or str(uuid.uuid4())

    start_time = time.perf_counter()

    try:
        result = pipeline.predict(payload.features)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected inference error: {str(error)}",
        ) from error

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    row = result.iloc[0]
    response = PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=float(row["fraud_probability"]),
        prediction=int(row["prediction"]),
        prediction_label=str(row["prediction_label"]),
        threshold=float(row["threshold"]),
        latency_ms=float(latency_ms),
        model_version=MODEL_VERSION,
    )

    write_audit_log(
        {
            "transaction_id": response.transaction_id,
            "model_version": response.model_version,
            "fraud_probability": response.fraud_probability,
            "prediction": response.prediction,
            "prediction_label": response.prediction_label,
            "threshold": response.threshold,
            "latency_ms": response.latency_ms,
        }
    )

    return response


def _explain_one(
    pipeline: FraudInferencePipeline,
    features: Dict[str, float],
    top_n: int,
) -> list[ExplanationItem]:
    X = pipeline.validate_and_align_features(features)
    dmatrix = xgb.DMatrix(X, feature_names=pipeline.feature_list)
    contributions = pipeline.model.get_booster().predict(
        dmatrix,
        pred_contribs=True,
    )

    shap_values = contributions[0][:-1]
    feature_values = X.iloc[0]
    ranked_indices = sorted(
        range(len(shap_values)),
        key=lambda index: abs(float(shap_values[index])),
        reverse=True,
    )[:top_n]

    explanations = []
    for index in ranked_indices:
        feature = pipeline.feature_list[index]
        shap_value = float(shap_values[index])
        if shap_value > 0:
            direction = "pushes toward Fraud"
        elif shap_value < 0:
            direction = "pushes toward Non-fraud"
        else:
            direction = "neutral"

        explanations.append(
            ExplanationItem(
                feature=feature,
                feature_value=float(feature_values[feature]),
                shap_value=shap_value,
                direction=direction,
            )
        )

    return explanations


def _predict_explain_one(
    payload: TransactionRequest,
    top_n: int,
) -> PredictionExplainResponse:
    pipeline = _get_pipeline()
    transaction_id = payload.transaction_id or str(uuid.uuid4())

    start_time = time.perf_counter()

    try:
        result = pipeline.predict(payload.features)
        top_explanations = _explain_one(pipeline, payload.features, top_n)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected explainability error: {str(error)}",
        ) from error

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    row = result.iloc[0]
    response = PredictionExplainResponse(
        transaction_id=transaction_id,
        fraud_probability=float(row["fraud_probability"]),
        prediction=int(row["prediction"]),
        prediction_label=str(row["prediction_label"]),
        threshold=float(row["threshold"]),
        latency_ms=float(latency_ms),
        model_version=MODEL_VERSION,
        top_explanations=top_explanations,
    )

    write_audit_log(
        {
            "endpoint": "predict-explain",
            "transaction_id": response.transaction_id,
            "model_version": response.model_version,
            "fraud_probability": response.fraud_probability,
            "prediction": response.prediction,
            "prediction_label": response.prediction_label,
            "threshold": response.threshold,
            "latency_ms": response.latency_ms,
            "top_explanation_features": [
                item.feature for item in response.top_explanations
            ],
        }
    )

    return response


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {
        "message": "Credit Card Fraud Detection API is running.",
        "docs": "Open /docs for Swagger UI.",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    pipeline = _get_pipeline()

    return HealthResponse(
        status="ok",
        model_loaded=pipeline.model is not None,
        threshold_loaded=pipeline.threshold is not None,
        feature_count=len(pipeline.feature_list),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    pipeline = _get_pipeline()

    return ModelInfoResponse(
        model_type=type(pipeline.model).__name__,
        model_version=MODEL_VERSION,
        threshold=float(pipeline.threshold),
        feature_count=len(pipeline.feature_list),
        input_type="model_ready_features",
        output_fields=[
            "transaction_id",
            "fraud_probability",
            "prediction",
            "prediction_label",
            "threshold",
            "latency_ms",
            "model_version",
        ],
    )


@app.get("/input-schema", tags=["Model"])
def input_schema() -> Dict[str, Any]:
    pipeline = _get_pipeline()

    return {
        "input_type": "model_ready_features",
        "number_of_required_features": len(pipeline.feature_list),
        "required_features": pipeline.feature_list,
        "note": (
            "The current API expects model-ready features. "
            "In real deployment, these features should be generated by a "
            "feature engineering layer before inference."
        ),
    }


@app.get(
    "/audit-summary",
    response_model=AuditSummaryResponse,
    tags=["Monitoring"],
)
def audit_summary() -> AuditSummaryResponse:
    """Summarize prediction audit-log activity for monitoring dashboards."""
    records = _load_audit_records()
    latencies = [
        float(record["latency_ms"])
        for record in records
        if isinstance(record.get("latency_ms"), (int, float))
    ]

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    latest_transaction_id = (
        str(records[-1].get("transaction_id"))
        if records and records[-1].get("transaction_id") is not None
        else None
    )

    return AuditSummaryResponse(
        total_requests=len(records),
        fraud_predictions=sum(1 for record in records if record.get("prediction") == 1),
        non_fraud_predictions=sum(
            1 for record in records if record.get("prediction") == 0
        ),
        mean_latency_ms=float(mean_latency),
        p95_latency_ms=_nearest_rank_percentile(latencies, 0.95),
        latest_transaction_id=latest_transaction_id,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: TransactionRequest) -> PredictionResponse:
    """Predict fraud probability for a single transaction."""
    return _predict_one(payload)


@app.post(
    "/predict-explain",
    response_model=PredictionExplainResponse,
    tags=["Prediction"],
)
def predict_explain(
    payload: TransactionRequest,
    top_n: int = Query(default=5, ge=1, le=20),
) -> PredictionExplainResponse:
    """Predict fraud probability and return top SHAP feature contributions."""
    return _predict_explain_one(payload, top_n)


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict fraud probability for multiple transactions."""
    results = [_predict_one(transaction) for transaction in payload.transactions]

    return BatchPredictionResponse(
        total_transactions=len(results),
        results=results,
    )
