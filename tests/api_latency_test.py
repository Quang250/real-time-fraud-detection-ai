"""Manual API latency test for repeated single-transaction predictions."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "reports" / "sample_transaction_payload.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "week7_api_latency_summary.csv"
BASE_URL = "http://127.0.0.1:8000"


def main():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    payload = {
        "transaction_id": "latency_test_tx",
        "features": features,
    }

    n_runs = 100
    latencies = []

    for _ in range(n_runs):
        start = time.perf_counter()

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=30,
        )

        end = time.perf_counter()

        if response.status_code != 200:
            print(response.status_code, response.text)
            continue

        latencies.append((end - start) * 1000)

    latency_summary = pd.DataFrame(
        [
            {
                "n_runs": len(latencies),
                "mean_latency_ms": np.mean(latencies),
                "median_latency_ms": np.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
            }
        ]
    )

    latency_summary.to_csv(OUTPUT_PATH, index=False)
    print(latency_summary)


if __name__ == "__main__":
    main()
