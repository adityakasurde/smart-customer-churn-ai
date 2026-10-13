"""Interactive Streamlit dashboard for Smart Customer Churn Prediction AI.

Provides single-customer churn risk inference, model performance comparison,
and optional prediction logging into a MySQL database.
"""

import sys
from pathlib import Path

# Add project root to sys.path to guarantee clean module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.ui_components import (
    DEFAULT_CUSTOMER_FORM_VALUES,
    FEATURE_OPTIONS,
    get_factual_interpretation,
    get_risk_badge_color,
    load_model_metadata,
    load_model_metrics,
)
from src.config import (
    CHAMPION_MODEL_NAME,
    DEFAULT_THRESHOLD,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
)
from src.database import (
    get_prediction_summary_stats,
    get_recent_prediction_logs,
    predict_and_log_customer,
    test_connection,
)
from src.predict import (
    PredictionEngineError,
    ValidationError,
    predict_single_customer,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Main Application Header & Overview
# -----------------------------------------------------------------------------
def render_overview_section():
    st.title("📊 Smart Customer Churn Prediction AI")
    st.markdown(
        "An industry-oriented machine learning system designed to estimate customer churn probability, "
        "stratify business risk tiers, and maintain prediction audit history."
    )

    metrics_data, metrics_err = load_model_metrics()
    metadata_data, _ = load_model_metadata()

    champion_name = CHAMPION_MODEL_NAME
    if metadata_data and "selected_champion" in metadata_data:
        champion_name = metadata_data.get("selected_champion", champion_name).replace("_", " ").title()

    if metrics_err or not metrics_data:
        st.warning(f"⚠️ Performance metrics notice: {metrics_err or 'Metrics file empty.'}")
        return

    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr_metrics = test_eval.get("logistic_regression", {})

    st.markdown("### 🏆 Champion Model Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Model Architecture", champion_name)
    with col2:
        st.metric("Test ROC-AUC", f"{lr_metrics.get('roc_auc', 0.8416):.4f}")
    with col3:
        st.metric("Churn Recall", f"{lr_metrics.get('recall', 0.7834):.1%}")
    with col4:
        st.metric("Precision", f"{lr_metrics.get('precision', 0.5043):.1%}")
    with col5:
        st.metric("F1-Score", f"{lr_metrics.get('f1_score', 0.6136):.4f}")

    st.caption(
        "ℹ️ **Evaluation Note:** Metrics represent offline evaluation on 1,409 held-out test customers, "
        "not real-time production accuracy. Churn is defined as positive class (1)."
    )
    st.divider()


# -----------------------------------------------------------------------------
# Customer Prediction Form
# -----------------------------------------------------------------------------
def render_prediction_section():
    st.subheader("🔍 Single Customer Churn Prediction")
    st.markdown(
        "Enter customer subscription, account, and demographic attributes below to estimate churn likelihood."
    )

    with st.form("churn_prediction_form"):
        col_dem, col_srv, col_bill = st.columns(3)

        with col_dem:
            st.markdown("##### 👤 Demographics & Tenure")
            gender = st.selectbox("Gender", FEATURE_OPTIONS["gender"], index=0)
            senior_raw = st.selectbox(
                "Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
            )
            partner = st.selectbox("Partner", FEATURE_OPTIONS["Partner"], index=1)
            dependents = st.selectbox("Dependents", FEATURE_OPTIONS["Dependents"], index=0)
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(DEFAULT_CUSTOMER_FORM_VALUES["tenure"]),
                step=1,
            )

        with col_srv:
            st.markdown("##### 🌐 Subscribed Services")
            phone_service = st.selectbox("Phone Service", FEATURE_OPTIONS["PhoneService"], index=0)
            multiple_lines = st.selectbox("Multiple Lines", FEATURE_OPTIONS["MultipleLines"], index=0)
            internet_service = st.selectbox("Internet Service", FEATURE_OPTIONS["InternetService"], index=0)
            online_security = st.selectbox("Online Security", FEATURE_OPTIONS["OnlineSecurity"], index=0)
            online_backup = st.selectbox("Online Backup", FEATURE_OPTIONS["OnlineBackup"], index=0)
            device_protection = st.selectbox("Device Protection", FEATURE_OPTIONS["DeviceProtection"], index=0)
            tech_support = st.selectbox("Tech Support", FEATURE_OPTIONS["TechSupport"], index=0)
            streaming_tv = st.selectbox("Streaming TV", FEATURE_OPTIONS["StreamingTV"], index=0)
            streaming_movies = st.selectbox("Streaming Movies", FEATURE_OPTIONS["StreamingMovies"], index=0)

        with col_bill:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox("Contract Type", FEATURE_OPTIONS["Contract"], index=0)
            paperless_billing = st.selectbox("Paperless Billing", FEATURE_OPTIONS["PaperlessBilling"], index=0)
            payment_method = st.selectbox("Payment Method", FEATURE_OPTIONS["PaymentMethod"], index=0)
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=250.0,
                value=float(DEFAULT_CUSTOMER_FORM_VALUES["MonthlyCharges"]),
                step=0.5,
                format="%.2f",
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=12000.0,
                value=float(DEFAULT_CUSTOMER_FORM_VALUES["TotalCharges"]),
                step=1.0,
                format="%.2f",
            )

            st.markdown("---")
            save_to_mysql = st.checkbox("💾 Save prediction to MySQL history", value=False)

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        customer_payload = {
            "gender": gender,
            "SeniorCitizen": senior_raw,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Executing prediction pipeline..."):
            try:
                if save_to_mysql:
                    result = predict_and_log_customer(customer_payload, save_to_db=True)
                else:
                    pred = predict_single_customer(customer_payload)
                    result = dict(pred)
                    result["logged_to_db"] = False
                    result["log_id"] = None
                    result["db_error"] = None

                st.session_state["latest_prediction"] = result
            except ValidationError as val_err:
                st.error(f"❌ Input validation failed: {val_err}")
                return
            except PredictionEngineError as pred_err:
                st.error(f"❌ Prediction pipeline failed: {pred_err}")
                return
            except Exception as exc:
                st.error(f"❌ Unexpected error occurred: {exc}")
                return

    # Render latest prediction result from session state if available
    if "latest_prediction" in st.session_state:
        res = st.session_state["latest_prediction"]
        prob = res["churn_probability"]
        prob_pct = res["churn_probability_percent"]
        pred_class = res["predicted_churn"]
        risk_cat = res["risk_category"]
        badge_color = get_risk_badge_color(risk_cat)

        st.markdown("#### 🎯 Prediction Results")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            st.metric("Churn Probability", f"{prob_pct:.2f}%")
        with res_col2:
            churn_label = "Yes (Likely to Churn)" if pred_class == 1 else "No (Likely to Stay)"
            st.metric("Predicted Churn", churn_label)
        with res_col3:
            st.markdown(
                f"<div style='background-color:{badge_color}; color:white; padding:8px 16px; border-radius:6px; "
                f"text-align:center; font-weight:bold; font-size:18px; margin-top:10px;'>{risk_cat} Risk</div>",
                unsafe_allow_html=True,
            )
        with res_col4:
            st.metric("Decision Threshold", f"{res.get('decision_threshold', DEFAULT_THRESHOLD)}")

        # Visual Probability Bar
        st.progress(float(prob))

        # Factual Domain Interpretation
        st.info(get_factual_interpretation(prob, risk_cat))

        # Database Logging Feedback
        if res.get("logged_to_db"):
            st.success(f"✅ Prediction successfully persisted to MySQL database (Record ID: #{res.get('log_id')}).")
        elif res.get("db_error"):
            st.warning(
                f"⚠️ Offline prediction completed. Database save was skipped: {res.get('db_error')}"
            )


# -----------------------------------------------------------------------------
# Prediction History & Analytics (MySQL Integration)
# -----------------------------------------------------------------------------
def render_history_section():
    st.subheader("📜 Historical Predictions & Aggregate KPIs")

    conn_ok, conn_msg = test_connection()

    if not conn_ok:
        st.info(
            "ℹ️ **Database Offline or Not Configured**: MySQL history is currently unavailable. "
            "The application operates in prediction-only mode without saving to database. "
            "To enable historical auditing, set valid MySQL credentials in your `.env` file and initialize the schema."
        )
        return

    # Controls
    h_col1, h_col2 = st.columns([4, 1])
    with h_col2:
        refresh = st.button("🔄 Refresh Data", use_container_width=True)

    try:
        stats = get_prediction_summary_stats()
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Logged Predictions", stats["total_predictions"])
        with kpi2:
            st.metric("High-Risk Alerts", stats["high_risk_predictions"])
        with kpi3:
            st.metric("Predicted Churn Count", stats["predicted_churn_count"])
        with kpi4:
            st.metric("Mean Churn Probability", f"{stats['avg_churn_probability']:.1%}")

        st.markdown("##### Recent Prediction Logs")
        recent_logs = get_recent_prediction_logs(limit=25)

        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)
            st.dataframe(
                df_logs[
                    [
                        "id",
                        "created_at",
                        "model_name",
                        "churn_probability",
                        "predicted_churn",
                        "risk_category",
                        "decision_threshold",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No prediction logs recorded yet. Submit a prediction with 'Save to MySQL' enabled.")
    except Exception as exc:
        st.warning(f"Failed to query database history: {exc}")


# -----------------------------------------------------------------------------
# Model Performance Comparison
# -----------------------------------------------------------------------------
def render_model_evaluation_section():
    st.subheader("📈 Model Performance & Evaluation Comparison")
    st.markdown(
        "Comparison of candidate models evaluated on the held-out test set (1,409 unseen customers, 20% stratified)."
    )

    metrics_data, metrics_err = load_model_metrics()
    if metrics_err or not metrics_data:
        st.warning(f"Evaluation metrics unavailable: {metrics_err}")
        return

    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr_m = test_eval.get("logistic_regression", {})
    rf_m = test_eval.get("random_forest", {})

    comparison_rows = [
        {
            "Metric": "ROC-AUC",
            "Logistic Regression (Champion)": f"{lr_m.get('roc_auc', 0):.4f}",
            "Random Forest": f"{rf_m.get('roc_auc', 0):.4f}",
            "Metric Focus": "Overall discrimination power across all thresholds",
        },
        {
            "Metric": "Recall (Caught Churners)",
            "Logistic Regression (Champion)": f"{lr_m.get('recall', 0):.1%} ({lr_m.get('confusion_matrix', {}).get('true_positives', 293)} / 374)",
            "Random Forest": f"{rf_m.get('recall', 0):.1%} ({rf_m.get('confusion_matrix', {}).get('true_positives', 283)} / 374)",
            "Metric Focus": "Proportion of actual churners caught (Telecom Priority)",
        },
        {
            "Metric": "Precision",
            "Logistic Regression (Champion)": f"{lr_m.get('precision', 0):.1%}",
            "Random Forest": f"{rf_m.get('precision', 0):.1%}",
            "Metric Focus": "Accuracy of churn alarms (Balances retention costs)",
        },
        {
            "Metric": "F1-Score",
            "Logistic Regression (Champion)": f"{lr_m.get('f1_score', 0):.4f}",
            "Random Forest": f"{rf_m.get('f1_score', 0):.4f}",
            "Metric Focus": "Harmonic mean of precision and recall",
        },
        {
            "Metric": "Accuracy",
            "Logistic Regression (Champion)": f"{lr_m.get('accuracy', 0):.1%}",
            "Random Forest": f"{rf_m.get('accuracy', 0):.1%}",
            "Metric Focus": "Overall correct predictions (Misleading under 73.5% imbalance)",
        },
    ]

    st.table(pd.DataFrame(comparison_rows))

    st.markdown(
        """
        **Model Selection Rationale:**
        - **Logistic Regression** was chosen as the champion model because it achieved stronger 5-fold cross-validation ROC-AUC (`0.8460` vs. `0.8450`) and higher churn recall (`80.13%` vs. `75.72%`) on training data, along with direct probability calibration and linear interpretability.
        - **Random Forest** achieved slightly higher raw test Accuracy (`75.37%` vs. `73.81%`) and F1-score (`0.6199` vs. `0.6136`) on the held-out test split, but missed 10 more actual churners (91 vs. 81 missed).
        - *Neither model strictly dominates on every single metric; Logistic Regression was selected to prioritize telecom churn recall.*
        """
    )


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main():
    render_overview_section()

    tabs = st.tabs(["🔮 Churn Prediction", "📜 Prediction History", "📊 Model Evaluation"])

    with tabs[0]:
        render_prediction_section()

    with tabs[1]:
        render_history_section()

    with tabs[2]:
        render_model_evaluation_section()


if __name__ == "__main__":
    main()
