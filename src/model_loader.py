"""Utilities for loading trained fraud detection artifacts."""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_fraud_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "best_threshold.pkl"
FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "feature_list.pkl"


def load_artifacts():
    """
    Load trained model, selected threshold, and ordered feature list.

    Returns
    -------
    model
        Trained fraud detection model.
    threshold
        Decision threshold for fraud classification.
    feature_list
        Ordered feature names used during model training.
    """
    missing_paths = [
        path
        for path in (MODEL_PATH, THRESHOLD_PATH, FEATURE_LIST_PATH)
        if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing model artifact(s):\n{missing}")

    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    feature_list = list(joblib.load(FEATURE_LIST_PATH))

    return model, threshold, feature_list
