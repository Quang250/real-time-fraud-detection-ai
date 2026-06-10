"""Manual API smoke test using the saved sample transaction payload."""

import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "reports" / "sample_transaction_payload.json"
BASE_URL = "http://127.0.0.1:8000"


def main():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    payload = {
        "transaction_id": "demo_tx_001",
        "features": features,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    print("Status code:", response.status_code)
    print(json.dumps(response.json(), indent=4))

    output_path = PROJECT_ROOT / "reports" / "week7_api_single_response.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(response.json(), file, indent=4)

    print(f"Saved response to: {output_path}")


if __name__ == "__main__":
    main()
