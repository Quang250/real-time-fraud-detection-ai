# Model Card - XGBoost Fraud Detection Model

## Model Overview

The final model is an XGBoost classifier trained for credit card fraud
detection. It predicts the probability that a transaction is fraudulent.

The model was selected after validation-set model comparison and threshold
tuning. The selected decision threshold is `0.73`.

## Dataset

The project uses a credit card fraud detection dataset with anonymized PCA
features (`V1` to `V28`) and engineered behavioral features created during the
feature engineering stage.

The dataset is highly imbalanced, with fraudulent transactions representing a
very small minority of all transactions. Because of this imbalance, model
evaluation focuses on precision, recall, F1-score, ROC-AUC, and PR-AUC rather
than accuracy alone.

## Input

The model expects `47` model-ready features generated from the feature
engineering stage. These include:

- anonymized PCA-transformed features,
- transaction amount and time features,
- engineered rolling/statistical amount features,
- engineered interaction features.

The current inference pipeline expects these model-ready features in the same
order as `models/feature_list.pkl`.

## Output

The inference pipeline returns:

- `fraud_probability`,
- `prediction`,
- `prediction_label`,
- `threshold`.

## Decision Threshold

The selected threshold is `0.73`, based on validation-set threshold tuning.

Predictions are generated as:

```text
prediction = 1 if fraud_probability >= 0.73 else 0
```

## Test Performance

Final test-set metrics:

- ROC-AUC: `0.9815`
- PR-AUC: `0.7892`
- Fraud Precision: `0.8889`
- Fraud Recall: `0.7467`
- Fraud F1-score: `0.8116`
- Accuracy: `0.9995`

Confusion matrix on the held-out test set:

- True Negative: `56880`
- False Positive: `7`
- False Negative: `19`
- True Positive: `56`

## Explainability

SHAP and LIME were used to explain the final XGBoost model.

SHAP was used as the primary explanation method because it supports global and
local interpretation for tree-based models. LIME was used as a complementary
local explanation method.

Important interpretation note: `V1` to `V28` are anonymized PCA-transformed
features and should not be interpreted as direct business variables.

## Limitations

- The dataset is anonymized, so PCA features cannot be directly interpreted as
  business variables.
- The current inference pipeline expects model-ready features, not raw
  transaction data.
- Real-time deployment is simulated at prototype level.
- False negatives still exist and require further monitoring and analysis.
- SHAP and LIME explain model behavior; they do not prove causal relationships.

## Intended Use

This model is intended for academic research and prototype demonstration of
fraud detection in a Fintech context.

It can be used to demonstrate:

- model evaluation on imbalanced fraud data,
- threshold-based fraud decisioning,
- explainable AI for financial risk analysis,
- preparation for a simulated real-time inference API.

## Not Intended For

This model should not be used as a production fraud detection system without:

- production-grade data validation,
- a complete feature engineering pipeline,
- monitoring for drift and false negatives,
- security and privacy review,
- business rule integration,
- human review workflows for high-risk decisions.

