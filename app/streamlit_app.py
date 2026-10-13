"""ChurnIQ — Smart Customer Churn AI: Production B2B SaaS Analytics Dashboard.

Enterprise-grade machine learning interface for single-customer churn risk inference,
model evaluation benchmarking, and prediction history auditing.
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
    CUSTOMER_PRESETS,
    DEFAULT_CUSTOMER_FORM_VALUES,
    FEATURE_OPTIONS,
    create_churn_gauge_chart,
    create_confusion_matrix_charts,
    create_metrics_comparison_chart,
    get_app_css,
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
    page_title="ChurnIQ — Smart Customer Churn AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom scoped CSS styling
st.markdown(get_app_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Application Header & Navigation Bar
# -----------------------------------------------------------------------------
def render_app_header(db_connected: bool):
    """Render the branded ChurnIQ top banner with system telemetry indicators."""
    db_status_pill = (
        '<span class="telemetry-pill"><span class="telemetry-dot-green"></span> Audit DB: Connected (MySQL 8.0)</span>'
        if db_connected
        else '<span class="telemetry-pill"><span class="telemetry-dot-slate"></span> Mode: Standalone (Cloud Ready)</span>'
    )

    st.markdown(
        f"""
        <div class="churniq-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 class="churniq-title">
                        ⚡ ChurnIQ <span class="churniq-badge">Enterprise ML v1.0.0</span>
                    </h1>
                    <div class="churniq-subtitle">Intelligent Customer Retention Analytics & Early Warning Platform</div>
                    <p class="churniq-desc">
                        Estimates telecom customer churn probability, quantifies attrition risk tiers, and delivers
                        actionable decision intelligence to maximize customer lifetime value (LTV).
                    </p>
                </div>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-pill"><span class="telemetry-dot-green"></span> Engine: Production Champion</span>
                <span class="telemetry-pill"><span class="telemetry-dot-blue"></span> Model: Logistic Regression (Balanced)</span>
                <span class="telemetry-pill"><span class="telemetry-dot-blue"></span> Offline Test ROC-AUC: 0.8416</span>
                <span class="telemetry-pill"><span class="telemetry-dot-blue"></span> Churn Recall: 78.34%</span>
                {db_status_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Compact & Useful Sidebar
# -----------------------------------------------------------------------------
def render_sidebar(db_connected: bool):
    """Render a compact, informative sidebar with telemetry and documentation."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h3 class="sidebar-brand-title">⚡ ChurnIQ</h3>
                <div class="sidebar-brand-sub">Customer Retention Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚙️ Production Telemetry")
        st.markdown(
            f"""
            - **Champion Architecture:** `Logistic Regression`
            - **Model Release:** `v1.0.0`
            - **Decision Threshold:** `{DEFAULT_THRESHOLD:.2f}`
            - **Held-Out Test ROC-AUC:** `0.8416`
            - **Evaluation Cohort:** `1,409 accounts (20%)`
            """
        )

        st.markdown("##### 🗄️ Infrastructure State")
        if db_connected:
            st.success("🟢 **MySQL Audit DB:** Connected (`smart_churn_ai`)")
        else:
            st.info("⚪ **Operating Mode:** Standalone / Cloud Ready (MySQL audit logging inactive)")

        st.markdown("##### 📚 Reference & Dataset")
        st.caption(
            "Trained on the **IBM Business Analytics Telco Customer Churn** dataset (7,043 total customer records, 19 predictive attributes)."
        )

        st.divider()
        st.caption("© 2026 ChurnIQ Analytics Inc. Built with Scikit-learn, Streamlit, and MySQL.")


# -----------------------------------------------------------------------------
# Section 1: Overview Dashboard
# -----------------------------------------------------------------------------
def render_overview_section():
    """Render the executive overview dashboard with verified champion metrics."""
    st.markdown('<div class="section-title">📊 Executive Overview & Champion Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Performance metrics for the selected production champion model evaluated on the held-out test split.</div>',
        unsafe_allow_html=True,
    )

    metrics_data, metrics_err = load_model_metrics()
    metadata_data, _ = load_model_metadata()

    champion_name = CHAMPION_MODEL_NAME
    if metadata_data and "selected_champion" in metadata_data:
        champion_name = metadata_data.get("selected_champion", champion_name).replace("_", " ").title()

    if metrics_err or not metrics_data:
        st.warning(f"⚠️ Performance metrics notice: {metrics_err or 'Metrics file is empty.'}")
        return

    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr_metrics = test_eval.get("logistic_regression", {})

    # Top KPI cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric(
            label="Champion Architecture",
            value=champion_name,
            help="Validated production pipeline with balanced class weighting",
        )
    with kpi_col2:
        st.metric(
            label="Held-Out Test ROC-AUC",
            value=f"{lr_metrics.get('roc_auc', 0.8416):.4f}",
            help="Measures overall discrimination capability across all classification thresholds",
        )
    with kpi_col3:
        st.metric(
            label="Churn Recall",
            value=f"{lr_metrics.get('recall', 0.7834):.1%}",
            help="Proportion of actual churners correctly caught (293 of 374)",
        )
    with kpi_col4:
        st.metric(
            label="Precision",
            value=f"{lr_metrics.get('precision', 0.5043):.1%}",
            help="Proportion of accounts flagged as churners that actually left",
        )
    with kpi_col5:
        st.metric(
            label="F1-Score",
            value=f"{lr_metrics.get('f1_score', 0.6136):.4f}",
            help="Harmonic mean balancing precision and recall",
        )

    # Metric Glossary & Strategic Alignment
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>ℹ️ Evaluation Disclaimer:</strong> Metrics reflect offline evaluation on <strong>1,409 held-out test customers</strong> 
            (20% stratified test split, unexposed during training), not real-time production accuracy. 
            Churn is defined as the positive class (<code>1</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="churniq-card">
                <div class="churniq-card-title">🎯 Why Recall is Priority #1</div>
                <p style="font-size:0.88rem; color:#475569; margin:0; line-height:1.5;">
                    In telecom, the financial cost of a <strong>False Negative</strong> (failing to detect an at-risk customer who then terminates) 
                    is severe: permanent loss of Customer Lifetime Value (LTV) and high acquisition costs to replace them. 
                    A <strong>False Positive</strong> merely prompts a proactive customer touchpoint or retention promotion.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="churniq-card">
                <div class="churniq-card-title">📐 ROC-AUC Discrimination</div>
                <p style="font-size:0.88rem; color:#475569; margin:0; line-height:1.5;">
                    The model achieves an offline <strong>ROC-AUC of 0.8416</strong>, confirming strong ranking power across probability cutoffs. 
                    This ensures customer risk rankings remain reliable whether targeting a broad retention campaign or focusing strictly on critical accounts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="churniq-card">
                <div class="churniq-card-title">🔍 Calibrated Probabilities</div>
                <p style="font-size:0.88rem; color:#475569; margin:0; line-height:1.5;">
                    Unlike uncalibrated tree ensembles that can output overconfident probabilities, Logistic Regression outputs mathematically well-calibrated posterior probabilities, enabling dependable business risk stratifications into Low, Medium, and High tiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Section 2: Customer Churn Prediction Experience
# -----------------------------------------------------------------------------
def render_prediction_section():
    """Render the 19-feature prediction form with quick-load presets and polished results."""
    st.markdown('<div class="section-title">🔮 Single Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Configure account attributes below or load a demonstration preset to estimate real-time churn probability.</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for form defaults if not already present
    if "form_preset_name" not in st.session_state:
        st.session_state["form_preset_name"] = "Default Profile (Baseline)"
    if "form_values" not in st.session_state:
        st.session_state["form_values"] = dict(DEFAULT_CUSTOMER_FORM_VALUES)

    # Preset Selector Bar
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        preset_names = list(CUSTOMER_PRESETS.keys())
        selected_preset = st.selectbox(
            "⚡ Quick-Fill Demonstration Profile",
            options=preset_names,
            index=preset_names.index(st.session_state["form_preset_name"])
            if st.session_state["form_preset_name"] in preset_names
            else 0,
            help="Instantly populate the 19 feature fields with realistic test scenarios to explore different risk tiers.",
        )
        if selected_preset != st.session_state["form_preset_name"]:
            st.session_state["form_preset_name"] = selected_preset
            st.session_state["form_values"] = dict(CUSTOMER_PRESETS[selected_preset])
            st.rerun()

    with p_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset to Baseline", use_container_width=True):
            st.session_state["form_preset_name"] = "Default Profile (Baseline)"
            st.session_state["form_values"] = dict(DEFAULT_CUSTOMER_FORM_VALUES)
            st.rerun()

    current_data = st.session_state.get("form_values", DEFAULT_CUSTOMER_FORM_VALUES)

    # 19-Feature Input Form
    with st.form("churn_prediction_form"):
        col_dem, col_srv, col_bill = st.columns(3)

        with col_dem:
            st.markdown("##### 👤 Customer Profile & Tenure")
            gender = st.selectbox(
                "Gender",
                FEATURE_OPTIONS["gender"],
                index=FEATURE_OPTIONS["gender"].index(current_data["gender"])
                if current_data["gender"] in FEATURE_OPTIONS["gender"]
                else 0,
            )
            senior_raw = st.selectbox(
                "Senior Citizen Status",
                [0, 1],
                index=int(current_data["SeniorCitizen"]),
                format_func=lambda x: "Yes (Senior Citizen)" if x == 1 else "No (Non-Senior)",
            )
            partner = st.selectbox(
                "Partner",
                FEATURE_OPTIONS["Partner"],
                index=FEATURE_OPTIONS["Partner"].index(current_data["Partner"])
                if current_data["Partner"] in FEATURE_OPTIONS["Partner"]
                else 0,
            )
            dependents = st.selectbox(
                "Dependents",
                FEATURE_OPTIONS["Dependents"],
                index=FEATURE_OPTIONS["Dependents"].index(current_data["Dependents"])
                if current_data["Dependents"] in FEATURE_OPTIONS["Dependents"]
                else 0,
            )
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(current_data["tenure"]),
                step=1,
                help="Total duration in months this customer has subscribed to services.",
            )

        with col_srv:
            st.markdown("##### 🌐 Subscribed Services")
            phone_service = st.selectbox(
                "Phone Service",
                FEATURE_OPTIONS["PhoneService"],
                index=FEATURE_OPTIONS["PhoneService"].index(current_data["PhoneService"])
                if current_data["PhoneService"] in FEATURE_OPTIONS["PhoneService"]
                else 0,
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                FEATURE_OPTIONS["MultipleLines"],
                index=FEATURE_OPTIONS["MultipleLines"].index(current_data["MultipleLines"])
                if current_data["MultipleLines"] in FEATURE_OPTIONS["MultipleLines"]
                else 0,
            )
            internet_service = st.selectbox(
                "Internet Service",
                FEATURE_OPTIONS["InternetService"],
                index=FEATURE_OPTIONS["InternetService"].index(current_data["InternetService"])
                if current_data["InternetService"] in FEATURE_OPTIONS["InternetService"]
                else 0,
            )
            online_security = st.selectbox(
                "Online Security",
                FEATURE_OPTIONS["OnlineSecurity"],
                index=FEATURE_OPTIONS["OnlineSecurity"].index(current_data["OnlineSecurity"])
                if current_data["OnlineSecurity"] in FEATURE_OPTIONS["OnlineSecurity"]
                else 0,
            )
            online_backup = st.selectbox(
                "Online Backup",
                FEATURE_OPTIONS["OnlineBackup"],
                index=FEATURE_OPTIONS["OnlineBackup"].index(current_data["OnlineBackup"])
                if current_data["OnlineBackup"] in FEATURE_OPTIONS["OnlineBackup"]
                else 0,
            )
            device_protection = st.selectbox(
                "Device Protection",
                FEATURE_OPTIONS["DeviceProtection"],
                index=FEATURE_OPTIONS["DeviceProtection"].index(current_data["DeviceProtection"])
                if current_data["DeviceProtection"] in FEATURE_OPTIONS["DeviceProtection"]
                else 0,
            )
            tech_support = st.selectbox(
                "Tech Support",
                FEATURE_OPTIONS["TechSupport"],
                index=FEATURE_OPTIONS["TechSupport"].index(current_data["TechSupport"])
                if current_data["TechSupport"] in FEATURE_OPTIONS["TechSupport"]
                else 0,
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                FEATURE_OPTIONS["StreamingTV"],
                index=FEATURE_OPTIONS["StreamingTV"].index(current_data["StreamingTV"])
                if current_data["StreamingTV"] in FEATURE_OPTIONS["StreamingTV"]
                else 0,
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                FEATURE_OPTIONS["StreamingMovies"],
                index=FEATURE_OPTIONS["StreamingMovies"].index(current_data["StreamingMovies"])
                if current_data["StreamingMovies"] in FEATURE_OPTIONS["StreamingMovies"]
                else 0,
            )

        with col_bill:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox(
                "Contract Type",
                FEATURE_OPTIONS["Contract"],
                index=FEATURE_OPTIONS["Contract"].index(current_data["Contract"])
                if current_data["Contract"] in FEATURE_OPTIONS["Contract"]
                else 0,
                help="Contract commitment terms. Month-to-month contracts exhibit the highest attrition.",
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                FEATURE_OPTIONS["PaperlessBilling"],
                index=FEATURE_OPTIONS["PaperlessBilling"].index(current_data["PaperlessBilling"])
                if current_data["PaperlessBilling"] in FEATURE_OPTIONS["PaperlessBilling"]
                else 0,
            )
            payment_method = st.selectbox(
                "Payment Method",
                FEATURE_OPTIONS["PaymentMethod"],
                index=FEATURE_OPTIONS["PaymentMethod"].index(current_data["PaymentMethod"])
                if current_data["PaymentMethod"] in FEATURE_OPTIONS["PaymentMethod"]
                else 0,
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=250.0,
                value=float(current_data["MonthlyCharges"]),
                step=0.5,
                format="%.2f",
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=12000.0,
                value=float(current_data["TotalCharges"]),
                step=1.0,
                format="%.2f",
            )

            st.markdown("<hr style='margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
            save_to_mysql = st.checkbox(
                "💾 Persist prediction to MySQL audit history",
                value=False,
                help="When enabled, saves the prediction result, probability, and risk tier to the prediction_logs table.",
            )

        submitted = st.form_submit_button(
            "⚡ Predict Churn Risk",
            type="primary",
            use_container_width=True,
        )

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

        with st.spinner("Executing inference pipeline..."):
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
                st.error(f"❌ Unexpected inference error: {exc}")
                return

    # Render persistent prediction results from session state
    if "latest_prediction" in st.session_state:
        res = st.session_state["latest_prediction"]
        prob = res["churn_probability"]
        prob_pct = res["churn_probability_percent"]
        pred_class = res["predicted_churn"]
        risk_cat = res["risk_category"]
        threshold = res.get("decision_threshold", DEFAULT_THRESHOLD)

        st.markdown("<hr style='margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Prediction Results & Retention Intelligence")

        # Top Metric Cards
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric(
                label="Churn Probability",
                value=f"{prob_pct:.2f}%",
                help="Estimated probability of contract termination",
            )
        with res_col2:
            churn_label = "Yes (Likely to Churn)" if pred_class == 1 else "No (Likely to Retain)"
            st.metric(
                label="Predicted Class",
                value=churn_label,
                help=f"Class classification evaluated at threshold {threshold:.2f}",
            )
        with res_col3:
            badge_class = (
                "risk-badge-high"
                if risk_cat == RISK_CATEGORY_HIGH
                else ("risk-badge-medium" if risk_cat == RISK_CATEGORY_MEDIUM else "risk-badge-low")
            )
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="text-align: center; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px;">
                    <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Assigned Risk Tier</div>
                    <span class="risk-badge {badge_class}">{risk_cat} Risk</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col4:
            st.metric(
                label="Decision Cutoff",
                value=f"{threshold:.2f}",
                help="Probability cutoff threshold separating non-churn from churn classification",
            )

        # Plotly Gauge Chart & Interpretation
        g_col, i_col = st.columns([1, 1])
        with g_col:
            st.plotly_chart(
                create_churn_gauge_chart(prob, risk_cat),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with i_col:
            st.markdown("##### 📋 Domain Interpretation")
            st.info(get_factual_interpretation(prob, risk_cat))

            # Strategic Retention Actions
            if risk_cat == RISK_CATEGORY_HIGH:
                st.markdown(
                    """
                    <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #991b1b;">
                        <strong>🚨 Recommended Retention Intervention:</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px;">
                            <li>Initiate proactive outreach within 48 hours.</li>
                            <li>Offer contract migration incentives (e.g. 1-year contract discount).</li>
                            <li>Review technical support satisfaction and bundle complementary security add-ons.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif risk_cat == RISK_CATEGORY_MEDIUM:
                st.markdown(
                    """
                    <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #92400e;">
                        <strong>⚠️ Recommended Retention Intervention:</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px;">
                            <li>Monitor customer health indicators and billing ticket activity.</li>
                            <li>Trigger automated email check-in highlighting unutilized service features.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #065f46;">
                        <strong>✅ Account Health: Stable</strong>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px;">
                            <li>Standard account maintenance. Customer shows strong commitment signals.</li>
                            <li>Eligible for premium tier upselling or annual subscription renewal bonuses.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Database Logging Feedback
        if res.get("logged_to_db"):
            st.success(
                f"✅ **Audit Trail Saved:** Prediction successfully persisted into MySQL database (`prediction_logs` record **#{res.get('log_id')}**)."
            )
        elif res.get("db_error"):
            st.warning(
                f"⚠️ **Offline Inference Preserved:** Model prediction completed, but database persistence failed: {res.get('db_error')}"
            )
        else:
            st.caption("ℹ️ *Prediction generated in standalone mode without persisting to database.*")


# -----------------------------------------------------------------------------
# Section 3: Prediction History & Analytics (MySQL Integration)
# -----------------------------------------------------------------------------
def render_history_section(db_connected: bool):
    """Render historical prediction analytics and the audit table with cloud fallbacks."""
    st.markdown('<div class="section-title">📜 Historical Predictions & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Audit history of persisted customer predictions logged into the MySQL database.</div>',
        unsafe_allow_html=True,
    )

    if not db_connected:
        st.markdown(
            """
            <div class="churniq-card" style="border-left: 4px solid #64748b;">
                <h4 style="margin-top:0; color:#0f172a;">⚪ Database Operating in Standalone / Cloud Mode</h4>
                <p style="font-size:0.9rem; color:#475569; margin-bottom:12px; line-height:1.5;">
                    Historical prediction auditing requires an active MySQL connection. In cloud deployments
                    (such as Streamlit Community Cloud) or local instances where MySQL is not running, 
                    <strong>ChurnIQ operates seamlessly in prediction-only mode</strong>.
                </p>
                <div style="background:#f1f5f9; padding:12px; border-radius:6px; font-size:0.83rem; color:#334155;">
                    <strong>To enable historical persistence:</strong>
                    <ol style="margin: 6px 0 0 0; padding-left: 20px;">
                        <li>Ensure a MySQL 8.0 server is reachable at your configured host.</li>
                        <li>Configure <code>MYSQL_HOST</code>, <code>MYSQL_PORT</code>, <code>MYSQL_DATABASE</code>, <code>MYSQL_USER</code>, and <code>MYSQL_PASSWORD</code> in your <code>.env</code> file.</li>
                        <li>Execute schema initialization: <code>python -c "from src.database import init_db; init_db()"</code>.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Top controls
    h_col1, h_col2 = st.columns([4, 1])
    with h_col2:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()

    try:
        stats = get_prediction_summary_stats()
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Predictions Logged", stats["total_predictions"])
        with kpi2:
            st.metric("High-Risk Alerts", stats["high_risk_predictions"])
        with kpi3:
            st.metric("Predicted Churn Count", stats["predicted_churn_count"])
        with kpi4:
            st.metric("Mean Churn Probability", f"{stats['avg_churn_probability']:.1%}")

        st.markdown("##### 📋 Recent Prediction Logs")
        recent_logs = get_recent_prediction_logs(limit=50)

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
                column_config={
                    "id": st.column_config.NumberColumn("Log ID", format="#%d"),
                    "created_at": st.column_config.TextColumn("Timestamp (UTC)"),
                    "model_name": st.column_config.TextColumn("Model"),
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn Probability",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "predicted_churn": st.column_config.NumberColumn("Predicted Churn", format="%d"),
                    "risk_category": st.column_config.TextColumn("Risk Tier"),
                    "decision_threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                },
            )
        else:
            st.info("No prediction logs recorded yet. Submit a prediction with 'Persist prediction to MySQL' checked.")
    except Exception as exc:
        st.warning(f"Failed to query database history: {exc}")


# -----------------------------------------------------------------------------
# Section 4: Model Evaluation & Benchmarks
# -----------------------------------------------------------------------------
def render_model_evaluation_section():
    """Render comprehensive model comparisons with interactive Plotly charts."""
    st.markdown('<div class="section-title">📈 Model Performance & Comparative Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Quantitative evaluation of candidate pipelines tested on the held-out test cohort (1,409 unseen customers, 20% stratified split).</div>',
        unsafe_allow_html=True,
    )

    metrics_data, metrics_err = load_model_metrics()
    if metrics_err or not metrics_data:
        st.warning(f"Evaluation metrics unavailable: {metrics_err}")
        return

    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr_m = test_eval.get("logistic_regression", {})
    rf_m = test_eval.get("random_forest", {})

    # Interactive Comparison Bar Chart
    st.markdown("##### 📊 Metric Comparison Across Candidate Pipelines")
    st.plotly_chart(
        create_metrics_comparison_chart(metrics_data),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Side-by-Side Confusion Matrices
    st.markdown("##### 🔲 Held-Out Confusion Matrices (Caught vs. Missed Churners)")
    st.plotly_chart(
        create_confusion_matrix_charts(metrics_data),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Structured Comparison Table
    st.markdown("##### 📑 Detailed Metric Breakdown")
    comparison_rows = [
        {
            "Metric": "ROC-AUC",
            "Logistic Regression (Champion)": f"{lr_m.get('roc_auc', 0):.4f}",
            "Random Forest": f"{rf_m.get('roc_auc', 0):.4f}",
            "Metric Focus": "Overall discrimination capability across all thresholds",
        },
        {
            "Metric": "Recall (Caught Churners)",
            "Logistic Regression (Champion)": f"{lr_m.get('recall', 0):.1%} ({lr_m.get('confusion_matrix', {}).get('true_positives', 293)} / 374)",
            "Random Forest": f"{rf_m.get('recall', 0):.1%} ({rf_m.get('confusion_matrix', {}).get('true_positives', 283)} / 374)",
            "Metric Focus": "Proportion of actual churners caught (Telecom Priority #1)",
        },
        {
            "Metric": "Precision",
            "Logistic Regression (Champion)": f"{lr_m.get('precision', 0):.1%}",
            "Random Forest": f"{rf_m.get('precision', 0):.1%}",
            "Metric Focus": "Accuracy of churn alarms (Controls retention campaign spend)",
        },
        {
            "Metric": "F1-Score",
            "Logistic Regression (Champion)": f"{lr_m.get('f1_score', 0):.4f}",
            "Random Forest": f"{rf_m.get('f1_score', 0):.4f}",
            "Metric Focus": "Harmonic mean balancing precision and recall",
        },
        {
            "Metric": "Accuracy",
            "Logistic Regression (Champion)": f"{lr_m.get('accuracy', 0):.1%}",
            "Random Forest": f"{rf_m.get('accuracy', 0):.1%}",
            "Metric Focus": "Overall correctness (Can be misleading under 73.5% non-churn imbalance)",
        },
    ]

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    # Strategic Selection Rationale
    st.markdown(
        """
        <div class="churniq-card" style="margin-top: 18px;">
            <div class="churniq-card-title">🏆 Champion Selection & Economic Rationale</div>
            <p style="font-size:0.9rem; color:#475569; line-height:1.6; margin-bottom:8px;">
                <strong>Logistic Regression</strong> was selected as the production champion for three rigorous reasons:
            </p>
            <ol style="font-size:0.88rem; color:#334155; line-height:1.6; margin-bottom:12px; padding-left: 22px;">
                <li><strong>Higher Churn Recall (78.34% vs. 75.67%):</strong> Logistic Regression caught <strong>293 actual churners</strong> versus 283 for Random Forest—missing 10 fewer departing customers on the held-out test set. In telecom economics, preserving customer lifetime value outweighs the cost of false positives.</li>
                <li><strong>Superior Cross-Validation Discrimination:</strong> Demonstrated stronger 5-fold CV ROC-AUC (<code>0.8460 ± 0.0124</code> vs. <code>0.8450 ± 0.0083</code>) during training split validation.</li>
                <li><strong>Calibrated Probabilities & Linear Interpretability:</strong> Direct posterior probabilities ensure that business risk stratification bands (0–30%, 30–60%, 60–100%) reflect actual frequencies.</li>
            </ol>
            <p style="font-size:0.85rem; color:#64748b; margin:0;">
                <em>Note: While Random Forest achieved slightly higher raw Accuracy (75.37% vs. 73.81%), accuracy is misleading in imbalanced churn data (where guessing "No Churn" yields 73.5% baseline accuracy).</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main():
    """Main application orchestrator."""
    # Check database connection state safely without crashing
    db_connected, _ = test_connection()

    # Render header & sidebar
    render_app_header(db_connected)
    render_sidebar(db_connected)

    # Clean 4-tab navigation pattern
    tabs = st.tabs(
        [
            "📊 Overview",
            "🔮 Churn Prediction",
            "📜 Prediction History",
            "📈 Model Evaluation",
        ]
    )

    with tabs[0]:
        render_overview_section()

    with tabs[1]:
        render_prediction_section()

    with tabs[2]:
        render_history_section(db_connected)

    with tabs[3]:
        render_model_evaluation_section()


if __name__ == "__main__":
    main()
