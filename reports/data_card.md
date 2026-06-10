# Data Card - Credit Card Fraud Detection Dataset

## Dataset Overview

This project uses the Credit Card Fraud Detection dataset from Kaggle. The
dataset contains anonymized credit card transactions and a binary fraud label.

## Dataset Size

- Total rows: `284,807`
- Fraud transactions: `492`
- Fraud rate: approximately `0.172%`

The dataset is severely imbalanced, so accuracy alone is not sufficient for
model evaluation.

## Features

The original dataset includes:

- `Time`
- `Amount`
- `V1` to `V28`
- `Class`

`V1` to `V28` are PCA-transformed anonymized features. They are useful for
modeling but cannot be directly interpreted as business variables such as
merchant, customer behavior, or device information.

## Target Variable

- `Class = 0`: normal transaction
- `Class = 1`: fraudulent transaction

## Project-Level Feature Engineering

The modeling pipeline expands the original dataset into `47` model-ready
features through time, amount, rolling-statistical, ratio, and interaction
features.

## Data Limitations

- No customer ID.
- No merchant ID.
- No device ID.
- No transaction location.
- PCA features are anonymized and cannot be directly explained as business
  concepts.
- The dataset is static and does not represent a live production transaction
  stream.

## Intended Use

This dataset is used for academic fraud detection modeling, explainable AI
experiments, and prototype API/dashboard demonstration.
