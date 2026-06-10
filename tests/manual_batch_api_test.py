"""Manual batch API smoke test using rows from the processed test set."""

import json
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_fe.csv"
BASE_URL = "http://127.0.0.1:8000"


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_sample = test_df.drop(columns=["Class"]).head(5)

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

    output_path = PROJECT_ROOT / "reports" / "week7_api_batch_response.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(response.json(), file, indent=4)

    print(f"Saved response to: {output_path}")


if __name__ == "__main__":
    main()
