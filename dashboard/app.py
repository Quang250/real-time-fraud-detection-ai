"""Streamlit dashboard for the fraud detection XAI demo."""

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from api_client import FraudAPIClient  # noqa: E402


st.set_page_config(
    page_title="Fraud Detection XAI Dashboard",
    page_icon=":credit_card:",
    layout="wide",
)

REPORTS_DIR = PROJECT_ROOT / "reports"
EXPLAIN_DIR = REPORTS_DIR / "explainability"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_single_sample(sample_type: str) -> tuple[dict[str, Any] | None, str, int | None]:
    test_path = DATA_DIR / "test_fe.csv"
    metadata_columns = {
        "actual",
        "predicted",
        "fraud_probability",
        "distance_to_threshold",
    }

    if sample_type == "Normal transaction":
        sample_features = load_json(REPORTS_DIR / "sample_transaction_payload.json")
        return sample_features, "dashboard_normal_tx", 0

    if sample_type == "Borderline transaction":
        borderline_path = EXPLAIN_DIR / "borderline_case.csv"
        if borderline_path.exists():
            borderline_df = pd.read_csv(borderline_path)
            if not borderline_df.empty:
                row = borderline_df.iloc[0]
                actual = int(row["actual"]) if "actual" in row else None
                features = row.drop(
                    labels=[col for col in metadata_columns if col in row.index]
                ).to_dict()
                return features, "dashboard_borderline_tx", actual

    if not test_path.exists():
        return None, "dashboard_missing_test_data", None

    test_df = pd.read_csv(test_path)

    if sample_type == "Fraud transaction":
        fraud_rows = test_df[test_df["Class"] == 1]
        if fraud_rows.empty:
            return None, "dashboard_missing_fraud_tx", None
        idx = fraud_rows.index[0]
        row = fraud_rows.loc[idx]
        return row.drop(labels=["Class"]).to_dict(), f"dashboard_fraud_tx_{idx}", 1

    idx = test_df.index[0]
    row = test_df.loc[idx]
    actual = int(row["Class"])
    return row.drop(labels=["Class"]).to_dict(), f"dashboard_borderline_tx_{idx}", actual


def show_file_status(label: str, path: Path) -> None:
    if path.exists():
        st.sidebar.success(f"{label}: found")
    else:
        st.sidebar.warning(f"{label}: missing")


def show_prediction_alert(prediction_label: str) -> None:
    if prediction_label == "Fraud":
        st.error("Fraudulent transaction detected")
    else:
        st.success("Transaction classified as Non-fraud")


def get_api_client(base_url: str) -> FraudAPIClient:
    return FraudAPIClient(base_url=base_url)


def call_api_safely(func: Callable, *args, **kwargs) -> tuple[Any, Any]:
    try:
        return func(*args, **kwargs), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to FastAPI server. Please start uvicorn first."
    except requests.exceptions.HTTPError as error:
        try:
            return None, error.response.json()
        except Exception:
            return None, str(error)
    except Exception as error:
        return None, str(error)


def nearest_threshold_row(threshold_df: pd.DataFrame, threshold: float) -> pd.Series:
    distance = (threshold_df["threshold"] - threshold).abs()
    return threshold_df.loc[distance.idxmin()]


st.sidebar.title("Dashboard Control")

api_base_url = st.sidebar.text_input(
    "FastAPI base URL",
    value="http://127.0.0.1:8000",
)

client = get_api_client(api_base_url)

st.sidebar.markdown("### Demo files")
show_file_status(
    "sample_transaction_payload.json",
    REPORTS_DIR / "sample_transaction_payload.json",
)
show_file_status("api_audit_log.jsonl", REPORTS_DIR / "api_audit_log.jsonl")
show_file_status("SHAP summary plot", EXPLAIN_DIR / "shap_summary_plot.png")
show_file_status("SHAP bar plot", EXPLAIN_DIR / "shap_bar_plot.png")

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard calls the FastAPI inference service and displays model, "
    "API, and XAI evidence."
)


st.title("Real-Time Credit Card Fraud Detection Dashboard")
st.markdown(
    """
    This dashboard demonstrates a Fintech fraud detection prototype using:

    - **XGBoost fraud detection model**
    - **FastAPI inference service**
    - **SHAP/LIME explainability**
    - **Audit logging**
    - **Near real-time transaction simulation**
    """
)

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


st.subheader("1. API Status")

health, health_error = call_api_safely(client.health)

if health_error:
    st.error("FastAPI server is not available.")
    st.write(health_error)
else:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("API Status", health.get("status", "unknown"))
    col2.metric("Model Loaded", str(health.get("model_loaded", False)))
    col3.metric("Threshold Loaded", str(health.get("threshold_loaded", False)))
    col4.metric("Feature Count", health.get("feature_count", 0))

    st.success("FastAPI service is running correctly.")


(
    tab_overview,
    tab_single,
    tab_batch,
    tab_threshold,
    tab_stream,
    tab_xai,
    tab_audit,
    tab_final,
) = st.tabs(
    [
        "Overview",
        "Single Prediction",
        "Batch Prediction",
        "Threshold Simulator",
        "Stream Simulation",
        "Explainability",
        "Audit & Monitoring",
        "Final Demo Package",
    ]
)


with tab_overview:
    st.header("Project Overview")

    model_info, model_error = call_api_safely(client.model_info)

    if model_error:
        st.warning("Cannot load model info from API.")
        st.write(model_error)
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Model", model_info.get("model_type", "N/A"))
        col2.metric("Version", model_info.get("model_version", "N/A"))
        col3.metric("Threshold", model_info.get("threshold", "N/A"))
        col4.metric("Features", model_info.get("feature_count", "N/A"))

        st.markdown("### Model information")
        st.json(model_info)

    st.markdown("### Technical pipeline")

    st.code(
        """
        Model-ready transaction
            ->
        Streamlit Dashboard
            ->
        FastAPI /predict endpoint
            ->
        FraudInferencePipeline
            ->
        XGBoost model + threshold 0.73
            ->
        Fraud probability + prediction label + latency + audit log
        """,
        language="text",
    )

    st.markdown("### Why this matters")
    st.write(
        """
        This project is not only a Kaggle-style classification notebook. It
        includes model training, threshold tuning, explainability, API
        deployment, logging, latency testing, and dashboard demonstration. This
        makes it closer to a real Fintech fraud detection prototype.
        """
    )


with tab_single:
    st.header("Single Transaction Prediction")

    sample_type = st.selectbox(
        "Sample type",
        ["Normal transaction", "Fraud transaction", "Borderline transaction"],
    )
    sample_features, default_transaction_id, actual_class = get_single_sample(sample_type)

    if sample_features is None:
        st.error("Sample transaction could not be loaded.")
    else:
        st.markdown("### Sample transaction")

        col1, col2 = st.columns(2)
        col1.metric("Selected sample", sample_type)
        col2.metric(
            "Actual class",
            "Unknown" if actual_class is None else actual_class,
        )

        with st.expander("View model-ready features"):
            st.json(sample_features)

        transaction_id = st.text_input(
            "Transaction ID",
            value=default_transaction_id,
        )
        top_n = st.slider(
            "Number of top explanations",
            min_value=3,
            max_value=10,
            value=5,
        )

        if st.button("Run single prediction"):
            result, error = call_api_safely(
                client.predict_explain,
                transaction_id,
                sample_features,
                top_n,
            )

            if error:
                st.error("Prediction failed.")
                st.write(error)
            else:
                st.success("Prediction successful.")
                show_prediction_alert(result["prediction_label"])

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability']:.6f}",
                )
                col2.metric("Prediction", result["prediction_label"])
                col3.metric("Threshold", result["threshold"])
                col4.metric("Latency", f"{result['latency_ms']:.2f} ms")

                st.markdown("### API response")
                st.json(result)

                if "top_explanations" in result:
                    st.markdown("### Top explanation features")
                    st.dataframe(pd.DataFrame(result["top_explanations"]))

                st.session_state["prediction_history"].append(result)

    if st.session_state["prediction_history"]:
        st.markdown("### Current session prediction history")
        st.dataframe(pd.DataFrame(st.session_state["prediction_history"]))


with tab_batch:
    st.header("Batch Prediction")

    test_path = DATA_DIR / "test_fe.csv"

    if not test_path.exists():
        st.error("test_fe.csv not found.")
    else:
        test_df = pd.read_csv(test_path)

        batch_mode = st.radio(
            "Batch mode",
            ["First N transactions", "Mixed normal + fraud transactions"],
            horizontal=True,
        )

        if batch_mode == "First N transactions":
            batch_size = st.slider(
                "Number of transactions for batch demo",
                min_value=1,
                max_value=20,
                value=5,
            )
            batch_sample = test_df.head(batch_size).copy()
        else:
            normal_rows = test_df[test_df["Class"] == 0].head(3)
            fraud_rows = test_df[test_df["Class"] == 1].head(2)
            batch_sample = pd.concat([normal_rows, fraud_rows]).sample(
                frac=1,
                random_state=42,
            )
            st.caption(
                "Mixed mode uses 3 normal transactions and 2 actual fraud "
                "transactions from the processed test set."
            )

        X_sample = batch_sample.drop(columns=["Class"])
        y_sample = batch_sample["Class"]

        transactions = []
        for idx, row in X_sample.iterrows():
            transactions.append(
                {
                    "transaction_id": f"dashboard_batch_tx_{idx}",
                    "features": row.to_dict(),
                }
            )

        st.markdown("### Batch input preview")
        st.dataframe(X_sample.head())

        actual_counts = y_sample.value_counts().rename(
            index={0: "Actual Non-fraud", 1: "Actual Fraud"}
        )
        col1, col2 = st.columns(2)
        col1.metric("Actual Non-fraud", int(actual_counts.get("Actual Non-fraud", 0)))
        col2.metric("Actual Fraud", int(actual_counts.get("Actual Fraud", 0)))

        if st.button("Run batch prediction"):
            result, error = call_api_safely(client.predict_batch, transactions)

            if error:
                st.error("Batch prediction failed.")
                st.write(error)
            else:
                st.success("Batch prediction successful.")

                results_df = pd.DataFrame(result["results"])
                results_df["actual"] = y_sample.values

                st.metric("Total transactions", result["total_transactions"])
                st.dataframe(results_df)

                fraud_count = int((results_df["prediction"] == 1).sum())
                non_fraud_count = int((results_df["prediction"] == 0).sum())

                if fraud_count > 0:
                    st.error(f"Batch contains {fraud_count} fraud prediction(s).")
                else:
                    st.success("Batch contains no fraud predictions.")

                col1, col2 = st.columns(2)
                col1.metric("Predicted Fraud", fraud_count)
                col2.metric("Predicted Non-fraud", non_fraud_count)


with tab_threshold:
    st.header("Threshold Simulator")

    threshold_path = REPORTS_DIR / "week8_threshold_simulation.csv"
    threshold_df = load_csv(threshold_path)

    if threshold_df.empty:
        st.warning("week8_threshold_simulation.csv not found or empty.")
    else:
        threshold_value = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.73,
            step=0.01,
        )
        selected_row = nearest_threshold_row(threshold_df, threshold_value)
        selected_threshold = float(selected_row["threshold"])

        st.caption(
            f"Showing nearest evaluated threshold: {selected_threshold:.2f}. "
            "Lower thresholds usually catch more fraud but create more false "
            "alerts. Higher thresholds reduce false alerts but may miss fraud."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{selected_row['precision']:.4f}")
        col2.metric("Recall", f"{selected_row['recall']:.4f}")
        col3.metric("F1-score", f"{selected_row['f1']:.4f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("False Positives", int(selected_row["fp"]))
        col2.metric("False Negatives", int(selected_row["fn"]))
        col3.metric(
            "Fraud Capture Rate",
            f"{selected_row['fraud_capture_rate'] * 100:.2f}%",
        )
        col4.metric(
            "False Alert Rate",
            f"{selected_row['false_alert_rate'] * 100:.4f}%",
        )

        st.markdown("### Risk trade-off curve")
        chart_df = threshold_df[
            ["threshold", "precision", "recall", "f1"]
        ].set_index("threshold")
        st.line_chart(chart_df)

        st.markdown("### Decision comparison")
        comparison_thresholds = [0.50, 0.73, 0.90]
        comparison_rows = []
        for item in comparison_thresholds:
            row = nearest_threshold_row(threshold_df, item)
            comparison_rows.append(
                {
                    "threshold": float(row["threshold"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "false_positives": int(row["fp"]),
                    "false_negatives": int(row["fn"]),
                    "fraud_capture_rate": float(row["fraud_capture_rate"]),
                    "false_alert_rate": float(row["false_alert_rate"]),
                }
            )

        st.dataframe(pd.DataFrame(comparison_rows))

        st.markdown("### Cost-based evaluation")
        st.caption(
            "Default business assumption: missing one fraud transaction costs "
            "100 units, while flagging one normal transaction costs 5 units."
        )

        col1, col2 = st.columns(2)
        fn_cost = col1.number_input(
            "Cost of one false negative",
            min_value=1,
            value=100,
            step=5,
        )
        fp_cost = col2.number_input(
            "Cost of one false positive",
            min_value=1,
            value=5,
            step=1,
        )

        cost_df = threshold_df.copy()
        cost_df["false_negative_cost"] = cost_df["fn"] * fn_cost
        cost_df["false_positive_cost"] = cost_df["fp"] * fp_cost
        cost_df["total_cost"] = (
            cost_df["false_negative_cost"] + cost_df["false_positive_cost"]
        )

        selected_cost_row = nearest_threshold_row(cost_df, threshold_value)
        best_cost_row = cost_df.loc[cost_df["total_cost"].idxmin()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Threshold Cost", f"{selected_cost_row['total_cost']:.0f}")
        col2.metric("Lowest Cost Threshold", f"{best_cost_row['threshold']:.2f}")
        col3.metric("Lowest Total Cost", f"{best_cost_row['total_cost']:.0f}")

        st.line_chart(cost_df.set_index("threshold")[["total_cost"]])

        cost_comparison = cost_df[
            [
                "threshold",
                "fp",
                "fn",
                "false_positive_cost",
                "false_negative_cost",
                "total_cost",
            ]
        ].sort_values("total_cost")

        st.markdown("### Lowest-cost thresholds")
        st.dataframe(cost_comparison.head(10))

        st.info(
            """
            Threshold selection is a risk-management decision. A lower threshold
            can catch more fraud, but it also increases false alerts. A higher
            threshold can reduce customer friction, but it may miss more fraud.
            """
        )


with tab_stream:
    st.header("Near Real-Time Stream Simulation")

    stream_path = REPORTS_DIR / "week7_api_stream_simulation.csv"
    stream_df = load_csv(stream_path)

    if stream_df.empty:
        st.warning("week7_api_stream_simulation.csv not found or empty.")
        st.write("Run the API stream simulation script first.")
    else:
        st.markdown("### Stream simulation result")
        st.dataframe(stream_df)

        total_rows = len(stream_df)

        if "actual" in stream_df.columns and "prediction" in stream_df.columns:
            correct = int((stream_df["actual"] == stream_df["prediction"]).sum())
            fraud_cases = int((stream_df["actual"] == 1).sum())
            predicted_fraud = int((stream_df["prediction"] == 1).sum())

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Transactions", total_rows)
            col2.metric("Correct", correct)
            col3.metric("Actual Fraud", fraud_cases)
            col4.metric("Predicted Fraud", predicted_fraud)

        if "api_latency_ms" in stream_df.columns:
            st.markdown("### API latency in stream")
            latency_summary = pd.DataFrame(
                [
                    {
                        "mean_latency_ms": stream_df["api_latency_ms"].mean(),
                        "median_latency_ms": stream_df["api_latency_ms"].median(),
                        "max_latency_ms": stream_df["api_latency_ms"].max(),
                    }
                ]
            )
            st.dataframe(latency_summary)


with tab_xai:
    st.header("Explainable AI Evidence")

    st.markdown(
        """
        This section shows how the model prediction can be explained. SHAP is
        used as the primary explanation method, while LIME is used as a
        complementary local explanation method.
        """
    )

    shap_summary = EXPLAIN_DIR / "shap_summary_plot.png"
    shap_bar = EXPLAIN_DIR / "shap_bar_plot.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SHAP Summary Plot")
        if shap_summary.exists():
            st.image(str(shap_summary), use_container_width=True)
        else:
            st.warning("SHAP summary plot not found.")

    with col2:
        st.subheader("SHAP Bar Plot")
        if shap_bar.exists():
            st.image(str(shap_bar), use_container_width=True)
        else:
            st.warning("SHAP bar plot not found.")

    st.markdown("### Explanation case summary")

    case_summary = load_csv(EXPLAIN_DIR / "explanation_case_summary.csv")

    if not case_summary.empty:
        st.dataframe(case_summary)
    else:
        st.warning("explanation_case_summary.csv not found.")

    st.markdown("### Local explanation tables")

    selected_case = st.selectbox(
        "Select explanation case",
        ["true_positive", "false_negative", "false_positive"],
    )

    shap_file_map = {
        "true_positive": "tp_shap_contributions.csv",
        "false_negative": "fn_shap_contributions.csv",
        "false_positive": "fp_shap_contributions.csv",
    }

    lime_file_map = {
        "true_positive": "lime_true_positive.csv",
        "false_negative": "lime_false_negative.csv",
        "false_positive": "lime_false_positive.csv",
    }

    shap_case_df = load_csv(EXPLAIN_DIR / shap_file_map[selected_case])
    lime_case_df = load_csv(EXPLAIN_DIR / lime_file_map[selected_case])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top SHAP contributions")
        if not shap_case_df.empty:
            st.dataframe(shap_case_df.head(10))
        else:
            st.warning("SHAP contribution file not found.")

    with col2:
        st.subheader("LIME explanation")
        if not lime_case_df.empty:
            st.dataframe(lime_case_df.head(10))
        else:
            st.warning("LIME explanation file not found.")

    st.info(
        """
        Important limitation: V1-V28 are anonymized PCA-transformed features.
        They should not be interpreted as direct business variables. SHAP
        values explain model behavior, not causal relationships.
        """
    )


with tab_audit:
    st.header("Audit Log and Monitoring")

    audit_summary, audit_summary_error = call_api_safely(client.audit_summary)

    if audit_summary_error:
        st.warning("Cannot load audit summary from API.")
        st.write(audit_summary_error)
    else:
        st.markdown("### Audit summary endpoint")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total prediction requests", audit_summary["total_requests"])
        col2.metric("Fraud predictions", audit_summary["fraud_predictions"])
        col3.metric("Non-fraud predictions", audit_summary["non_fraud_predictions"])

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Average latency",
            f"{audit_summary['mean_latency_ms']:.2f} ms",
        )
        col2.metric("P95 latency", f"{audit_summary['p95_latency_ms']:.2f} ms")
        col3.metric(
            "Latest transaction ID",
            audit_summary.get("latest_transaction_id") or "N/A",
        )

    audit_path = REPORTS_DIR / "api_audit_log.jsonl"
    audit_df = load_jsonl(audit_path)

    if audit_df.empty:
        st.warning("api_audit_log.jsonl not found or empty.")
    else:
        st.markdown("### Audit log preview")
        st.dataframe(audit_df.tail(20))

        col1, col2, col3 = st.columns(3)

        col1.metric("Total log rows", len(audit_df))

        if "prediction" in audit_df.columns:
            col2.metric(
                "Fraud predictions",
                int((audit_df["prediction"] == 1).sum()),
            )
            col3.metric(
                "Non-fraud predictions",
                int((audit_df["prediction"] == 0).sum()),
            )

        if "latency_ms" in audit_df.columns:
            st.markdown("### Latency summary from audit log")
            latency_df = pd.DataFrame(
                [
                    {
                        "mean_latency_ms": audit_df["latency_ms"].mean(),
                        "median_latency_ms": audit_df["latency_ms"].median(),
                        "p95_latency_ms": audit_df["latency_ms"].quantile(0.95),
                        "max_latency_ms": audit_df["latency_ms"].max(),
                    }
                ]
            )
            st.dataframe(latency_df)

    st.markdown("### Why audit log matters")

    st.write(
        """
        Audit logging supports traceability. If a transaction is flagged or
        missed, the system can later review the transaction ID, model version,
        fraud probability, threshold, prediction label, and latency.
        """
    )


with tab_final:
    st.header("Final Demo Package")

    st.markdown("### Evidence checklist")

    checklist = pd.DataFrame(
        [
            {
                "item": "FastAPI server running",
                "evidence": "Swagger UI /docs",
                "status": "Done",
            },
            {
                "item": "Health endpoint",
                "evidence": "GET /health returns 200",
                "status": "Done",
            },
            {
                "item": "Model info endpoint",
                "evidence": "GET /model-info returns model metadata",
                "status": "Done",
            },
            {
                "item": "Single prediction",
                "evidence": "week7_predict_success_response.json",
                "status": "Done",
            },
            {
                "item": "Batch prediction",
                "evidence": "week7_predict_batch_success_response.json",
                "status": "Done",
            },
            {
                "item": "Input validation",
                "evidence": "422 response when features are missing",
                "status": "Done",
            },
            {
                "item": "Latency test",
                "evidence": "week7_api_latency_summary.csv",
                "status": "Done",
            },
            {
                "item": "Stream simulation",
                "evidence": "week7_api_stream_simulation.csv",
                "status": "Done",
            },
            {
                "item": "Explainability",
                "evidence": "SHAP/LIME reports",
                "status": "Done",
            },
            {
                "item": "Audit log",
                "evidence": "api_audit_log.jsonl",
                "status": "Done",
            },
        ]
    )

    st.dataframe(checklist)

    st.markdown("### Suggested demo order")

    st.code(
        """
        1. Open FastAPI Swagger UI and show endpoints.
        2. Open Streamlit dashboard.
        3. Show API status and model information.
        4. Run single prediction.
        5. Run batch prediction.
        6. Show stream simulation.
        7. Show SHAP/LIME explainability evidence.
        8. Show audit log and latency.
        9. Explain limitations and future improvements.
        """,
        language="text",
    )

    st.markdown("### Limitations")
    st.write(
        """
        - API currently receives model-ready features, not raw transaction data.
        - PCA features cannot be directly interpreted as business variables.
        - Dataset does not include customer ID, merchant ID, device, or location.
        - Real-time deployment is simulated at prototype level.
        - False negatives still exist and require monitoring.
        """
    )

    st.markdown("### Future Work")
    st.write(
        """
        - Add a feature store or real-time feature engineering layer.
        - Add data drift and model drift monitoring.
        - Add a fraud analyst feedback loop.
        - Add a model retraining pipeline.
        - Deploy the API using Docker or a cloud service.
        """
    )

    st.markdown("### Key message for thesis defense")

    st.success(
        """
        This project goes beyond a standard Kaggle notebook by integrating
        machine learning, threshold tuning, explainability, FastAPI deployment,
        audit logging, latency testing, stream simulation, and dashboard
        visualization.
        """
    )
