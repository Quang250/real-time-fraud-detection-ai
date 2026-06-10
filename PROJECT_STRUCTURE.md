# Project Structure

## api/

Contains the FastAPI inference service.

Main entrypoint:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Important files:

- `main.py`
- `schemas.py`
- `audit_logger.py`

## dashboard/

Contains the Streamlit dashboard.

Main entrypoint:

```bash
streamlit run dashboard/app.py
```

## src/

Contains reusable machine learning and inference code.

Important files:

- `model_loader.py`
- `inference_pipeline.py`
- `batch_inference.py`
- `validate_artifacts.py`
- `feature_engineering.py`
- `train_model.py`
- `evaluate_model.py`
- `explainability.py`

## notebooks/

Contains the weekly experiment notebooks:

- `01_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_modeling.ipynb`
- `04_explainability.ipynb`
- `05_pipeline_packaging.ipynb`

## reports/

Contains metrics, plots, API responses, explainability outputs, screenshots,
model documentation, and final demo evidence.

## tests/

Contains automated API tests and manual testing scripts.

## data/

Contains data instructions and optional small sample files.

The full raw dataset and processed train/test files are not committed.

## models/

Stores local model artifacts after training.

Model artifacts are not committed by default.
