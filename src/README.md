# Source Code Map

This folder contains the reusable Python code for the fraud detection
project. The current production-like scope is Week 6 inference packaging.

## Active Week 6 Pipeline

- `model_loader.py`: loads the saved XGBoost model, decision threshold, and
  ordered feature list from `models/`.
- `inference_pipeline.py`: validates model-ready input features, aligns them
  to the training feature order, and returns fraud predictions.

## Utility Entrypoints

- `validate_artifacts.py`: command-line check for model artifact loading.
  Run from the project root with `python -m src.validate_artifacts`.
- `batch_inference.py`: batch CSV inference helper using
  `FraudInferencePipeline`.

## Placeholder Modules

The files below are kept as planned codebase sections, but they are not part
of the active Week 6 inference pipeline yet:

- `data_preprocessing.py`
- `feature_engineering.py`
- `train_model.py`
- `evaluate_model.py`
- `explainability.py`
- `inference_api.py`

For the current report, offline EDA, feature engineering, modeling, and XAI
logic still live mainly in notebooks. These placeholder modules should not be
presented as completed production code.
