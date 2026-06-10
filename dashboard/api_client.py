"""Client wrapper for the FastAPI fraud detection service."""

from typing import Any, Dict, List

import requests


class FraudAPIClient:
    """Simple API client for the FastAPI fraud detection service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=30)
        response.raise_for_status()
        return response.json()

    def model_info(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/model-info", timeout=30)
        response.raise_for_status()
        return response.json()

    def input_schema(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/input-schema", timeout=30)
        response.raise_for_status()
        return response.json()

    def audit_summary(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/audit-summary", timeout=30)
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        transaction_id: str,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        payload = {
            "transaction_id": transaction_id,
            "features": features,
        }

        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def predict_explain(
        self,
        transaction_id: str,
        features: Dict[str, float],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        payload = {
            "transaction_id": transaction_id,
            "features": features,
        }

        response = requests.post(
            f"{self.base_url}/predict-explain",
            params={"top_n": top_n},
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "transactions": transactions,
        }

        response = requests.post(
            f"{self.base_url}/predict-batch",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()
