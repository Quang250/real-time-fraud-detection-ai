"""Pydantic request and response schemas for the fraud detection API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Request schema for a single model-ready transaction."""

    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional transaction ID for tracking and audit logging.",
    )
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Dictionary of model-ready features. Must contain all required features."
        ),
    )


class PredictionResponse(BaseModel):
    """Response schema for a single fraud prediction."""

    transaction_id: str
    fraud_probability: float
    prediction: int
    prediction_label: str
    threshold: float
    latency_ms: float
    model_version: str


class ExplanationItem(BaseModel):
    """One feature-level explanation item for a prediction."""

    feature: str
    feature_value: float
    shap_value: float
    direction: str


class PredictionExplainResponse(PredictionResponse):
    """Prediction response enriched with top feature explanations."""

    top_explanations: List[ExplanationItem]


class BatchPredictionRequest(BaseModel):
    """Request schema for multiple transactions."""

    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    total_transactions: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint."""

    status: str
    model_loaded: bool
    threshold_loaded: bool
    feature_count: int


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    model_type: str
    model_version: str
    threshold: float
    feature_count: int
    input_type: str
    output_fields: List[str]


class AuditSummaryResponse(BaseModel):
    """Response schema for prediction audit-log summary."""

    total_requests: int
    fraud_predictions: int
    non_fraud_predictions: int
    mean_latency_ms: float
    p95_latency_ms: float
    latest_transaction_id: Optional[str]
