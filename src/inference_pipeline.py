"""Reusable inference pipeline for fraud detection."""

import numpy as np
import pandas as pd

from src.model_loader import load_artifacts


class FraudInferencePipeline:
    """
    Inference pipeline for credit card fraud detection.

    The pipeline expects model-ready features with the same columns used
    when the model was trained. Feature engineering is intentionally kept
    outside this class so it can be reused by API and dashboard layers.
    """

    def __init__(self):
        self.model, self.threshold, self.feature_list = load_artifacts()

    def validate_and_align_features(self, data):
        """
        Validate input data and align columns to the training feature order.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            One transaction as a dictionary, or many transactions as a
            DataFrame. The input must contain all model-ready features.

        Returns
        -------
        pandas.DataFrame
            Cleaned feature matrix ordered according to ``feature_list``.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dict or pandas DataFrame.")

        if df.empty:
            raise ValueError("Input data must contain at least one row.")

        missing_features = [
            feature
            for feature in self.feature_list
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        extra_features = [
            column
            for column in df.columns
            if column not in self.feature_list
        ]

        if extra_features:
            df = df.drop(columns=extra_features)

        df = df[self.feature_list]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        return df

    def predict(self, data):
        """
        Predict fraud probability and label for one or more transactions.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Model-ready transaction features.

        Returns
        -------
        pandas.DataFrame
            Prediction result with fraud probability, binary prediction,
            human-readable label, and decision threshold.
        """
        X = self.validate_and_align_features(data)

        fraud_probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (fraud_probabilities >= self.threshold).astype(int)

        return pd.DataFrame(
            {
                "fraud_probability": fraud_probabilities,
                "prediction": predictions,
                "prediction_label": [
                    "Fraud" if pred == 1 else "Non-fraud"
                    for pred in predictions
                ],
                "threshold": self.threshold,
            },
            index=X.index,
        ).reset_index(drop=True)
