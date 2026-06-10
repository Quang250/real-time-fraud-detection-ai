# Data

This project uses the Credit Card Fraud Detection dataset from Kaggle.

## Dataset

The original dataset contains anonymized credit card transactions with
PCA-transformed features `V1` to `V28`, plus `Time`, `Amount`, and `Class`.

## Why Raw Data Is Not Included

The full raw dataset is not committed to this repository because it is large and
should be downloaded from Kaggle by each user who wants to reproduce the full
workflow.

## How To Reproduce

1. Download the Credit Card Fraud Detection dataset from Kaggle.
2. Place `creditcard.csv` into:

```text
data/raw/creditcard.csv
```

3. Run the feature engineering notebook or script to generate processed files.

## Expected Local Data Structure

```text
data/
|-- raw/
|   `-- creditcard.csv
|-- processed/
|   |-- train_fe.csv
|   `-- test_fe.csv
`-- sample/
    `-- sample_transaction_payload.json
```

Only small sample files are intended to be committed.
