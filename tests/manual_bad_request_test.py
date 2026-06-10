"""Manual API validation test with one required feature intentionally missing."""

import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "reports" / "sample_transaction_payload.json"
BASE_URL = "http://127.0.0.1:8000"


def main():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    # Remove one required feature intentionally.
    features.pop("Time", None)

    payload = {
        "transaction_id": "bad_tx_missing_time",
        "features": features,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    print("Status code:", response.status_code)
    print(json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    main()
