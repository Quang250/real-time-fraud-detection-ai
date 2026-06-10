# Quick Start

## 1. Create Environment

```bash
conda env create -f environment.yml
conda activate fraud-xai
```

If the environment already exists:

```bash
conda activate fraud-xai
pip install -r requirements.txt
```

## 2. Prepare Data

Download the Kaggle Credit Card Fraud Detection dataset and place it at:

```text
data/raw/creditcard.csv
```

Then run the notebooks in order to reproduce processed data, model artifacts,
and reports.

## 3. Validate Artifacts

```bash
python -m src.validate_artifacts
```

## 4. Run FastAPI

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 5. Run Streamlit Dashboard

Open a second terminal:

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

## 6. Run Tests

```bash
pytest tests/test_api.py
```
