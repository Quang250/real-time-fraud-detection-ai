# Final Demo Script - Fraud Detection XAI System

## 1. Opening

This project develops a credit card fraud detection prototype using machine
learning, explainable AI, FastAPI, and Streamlit. The goal is to simulate how a
Fintech system can detect suspicious transactions and explain model decisions.

## 2. Model Summary

The final model is XGBoost with a selected decision threshold of 0.73. The model
was evaluated using PR-AUC, ROC-AUC, Precision, Recall, and F1-score due to
severe class imbalance.

## 3. API Layer

The trained model was packaged into an inference pipeline and exposed through
FastAPI. The API supports health check, model information, input schema, single
prediction, prediction with top SHAP explanations, and batch prediction.

## 4. Dashboard Layer

The Streamlit dashboard connects to the FastAPI service and displays model
prediction results, batch prediction results, stream simulation, explainability
outputs, audit logs, and latency evidence.

## 5. Explainability

SHAP and LIME were used to explain model predictions. SHAP provides global and
local explanations, while LIME serves as a complementary local explanation
method.

## 6. Monitoring and Audit

Each prediction request is recorded in an audit log with transaction ID, model
version, fraud probability, prediction label, threshold, and latency.

## 7. Limitation

The current prototype uses model-ready features rather than raw transaction
data. In a real-world system, a feature engineering layer would be needed before
inference.

## 8. Closing Message

The main contribution of this project is the integration of fraud detection
modeling, explainability, API deployment, audit logging, latency testing, stream
simulation, and dashboard visualization into one coherent Fintech prototype.
