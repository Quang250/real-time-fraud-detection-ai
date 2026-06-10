"""Manual /predict-batch test using processed test-set sample rows."""

import json
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_fe.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "week7_predict_batch_success_response.json"
BASE_URL = "http://127.0.0.1:8000"


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_sample = test_df.drop(columns=["Class"]).head(3)

    transactions = []

    for idx, row in X_sample.iterrows():
        transactions.append(
            {
                "transaction_id": f"batch_tx_{idx}",
                "features": row.to_dict(),
            }
        )

    payload = {
        "transactions": transactions,
    }

    response = requests.post(
        f"{BASE_URL}/predict-batch",
        json=payload,
        timeout=30,
    )

    print("Status code:", response.status_code)
    print(json.dumps(response.json(), indent=4))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(response.json(), file, indent=4)

    print(f"Saved response to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
