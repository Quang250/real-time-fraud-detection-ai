"""Batch inference entrypoint for model-ready fraud features."""

import pandas as pd

from src.inference_pipeline import FraudInferencePipeline


def run_batch_inference(
    input_csv_path,
    output_csv_path,
    label_column="Class",
):
    """
    Run batch fraud inference on a CSV file.

    Parameters
    ----------
    input_csv_path : str
        Path to input CSV file.
    output_csv_path : str
        Path to output CSV file.
    label_column : str
        Optional target column name if labels are available.

    Returns
    -------
    pandas.DataFrame
        Original input plus prediction columns.
    """
    df = pd.read_csv(input_csv_path)
    pipeline = FraudInferencePipeline()

    if label_column in df.columns:
        X = df.drop(columns=[label_column])
        y = df[label_column]
    else:
        X = df
        y = None

    predictions = pipeline.predict(X)

    output_df = df.copy()
    output_df["fraud_probability"] = predictions["fraud_probability"].values
    output_df["prediction"] = predictions["prediction"].values
    output_df["prediction_label"] = predictions["prediction_label"].values

    if y is not None:
        output_df["actual"] = y.values

    output_df.to_csv(output_csv_path, index=False)

    return output_df


if __name__ == "__main__":
    output = run_batch_inference(
        input_csv_path="data/processed/test_fe.csv",
        output_csv_path="reports/week6_batch_prediction_test.csv",
    )

    print(output.head())
