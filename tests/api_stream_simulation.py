"""Mixed stream simulation through the FastAPI prediction endpoint."""

import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_fe.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "week7_api_stream_simulation.csv"
BASE_URL = "http://127.0.0.1:8000"


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)

    fraud_rows = test_df[test_df["Class"] == 1].head(5)
    normal_rows = test_df[test_df["Class"] == 0].head(5)

    stream_df = pd.concat([normal_rows, fraud_rows]).sample(
        frac=1,
        random_state=42,
    )

    results = []

    for idx, row in stream_df.iterrows():
        actual = int(row["Class"])
        features = row.drop(labels=["Class"]).to_dict()

        payload = {
            "transaction_id": f"stream_tx_{idx}",
            "features": features,
        }

        start = time.perf_counter()

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=30,
        )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        response_json = response.json()

        results.append(
            {
                "transaction_index": idx,
                "actual": actual,
                "fraud_probability": response_json.get("fraud_probability"),
                "prediction": response_json.get("prediction"),
                "prediction_label": response_json.get("prediction_label"),
                "threshold": response_json.get("threshold"),
                "api_latency_ms": latency_ms,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(results_df)


if __name__ == "__main__":
    main()
