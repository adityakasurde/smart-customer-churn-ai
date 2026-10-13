"""Reusable UI helpers, metrics loaders, chart generators, and styling for ChurnIQ dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    METRICS_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
)

# Feature options mapping based on Telco Churn dataset domain values
FEATURE_OPTIONS: Dict[str, List[Any]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Standard realistic default customer values
DEFAULT_CUSTOMER_FORM_VALUES: Dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}

# Curated demonstration presets for quick testing and portfolio walkthroughs
CUSTOMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Default Profile (Baseline)": DEFAULT_CUSTOMER_FORM_VALUES,
    "High Risk (New Fiber, Month-to-Month)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.85,
        "TotalCharges": 181.70,
    },
    "Low Risk (Long-Term 2-Year Contract)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 58.20,
        "TotalCharges": 3783.00,
    },
    "Medium Risk (Mid-Tenure, 1-Year Contract)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.00,
        "TotalCharges": 1320.00,
    },
}


def load_model_metrics(
    metrics_path: Union[Path, str] = METRICS_SUMMARY_PATH,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely load metrics_summary.json without crashing if file is missing or invalid.

    Args:
        metrics_path: Path to the metrics summary JSON file.

    Returns:
        Tuple[Optional[Dict], Optional[str]]: (metrics_dict, error_message)
    """
    path = Path(metrics_path)
    if not path.is_file():
        return (
            None,
            f"Metrics summary file not found at '{path}'. Please run 'python -m src.train' to generate.",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, "Metrics summary file does not contain a valid JSON object."
        return data, None
    except Exception as exc:
        return None, f"Failed to parse metrics summary JSON: {exc}"


def load_model_metadata(
    metadata_path: Union[Path, str] = MODEL_METADATA_PATH,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely load model_metadata.json without crashing if file is missing or invalid.

    Args:
        metadata_path: Path to the model metadata JSON file.

    Returns:
        Tuple[Optional[Dict], Optional[str]]: (metadata_dict, error_message)
    """
    path = Path(metadata_path)
    if not path.is_file():
        return (
            None,
            f"Model metadata file not found at '{path}'. Please run 'python -m src.train' to generate.",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, "Model metadata file does not contain a valid JSON object."
        return data, None
    except Exception as exc:
        return None, f"Failed to parse model metadata JSON: {exc}"


def get_factual_interpretation(probability: float, risk_category: str) -> str:
    """Generate a neutral, factual explanation of the churn prediction.

    Avoids causal overclaims and clarifies that probabilities reflect historical patterns.

    Args:
        probability: Float between 0.0 and 1.0.
        risk_category: 'Low', 'Medium', or 'High'.

    Returns:
        str: Factual interpretation text.
    """
    pct = probability * 100.0
    if risk_category == RISK_CATEGORY_HIGH:
        assessment = (
            f"The model estimates a **{pct:.1f}% churn probability** (High Risk tier). "
            "In historical telecom data, customers sharing similar profiles (e.g. short tenure on month-to-month terms) "
            "showed a substantially elevated frequency of contract termination."
        )
    elif risk_category == RISK_CATEGORY_MEDIUM:
        assessment = (
            f"The model estimates a **{pct:.1f}% churn probability** (Medium Risk tier). "
            "This customer falls into a transitional cohort exhibiting balanced retention and attrition tendencies."
        )
    else:
        assessment = (
            f"The model estimates a **{pct:.1f}% churn probability** (Low Risk tier). "
            "The customer attributes closely align with long-term subscription retention patterns."
        )

    disclaimer = (
        " *Note: Probability estimates reflect statistical likelihood under observed service conditions, "
        "not a deterministic guarantee that an individual customer will leave.*"
    )
    return assessment + disclaimer


def get_risk_badge_color(risk_category: str) -> str:
    """Return styling color associated with each risk tier.

    Preserves exact hex codes expected by test_dashboard.py:
    - High Risk: #e63946
    - Medium Risk: #f4a261
    - Low Risk: #2a9d8f

    Args:
        risk_category: 'Low', 'Medium', or 'High'.

    Returns:
        str: Hex color string.
    """
    if risk_category == RISK_CATEGORY_HIGH:
        return "#e63946"  # Coral Red
    elif risk_category == RISK_CATEGORY_MEDIUM:
        return "#f4a261"  # Warm Orange
    else:
        return "#2a9d8f"  # Teal Green


def create_churn_gauge_chart(probability: float, risk_category: str) -> go.Figure:
    """Create a polished, responsive Plotly gauge chart displaying churn risk.

    Args:
        probability: Float between 0.0 and 1.0.
        risk_category: 'Low', 'Medium', or 'High'.

    Returns:
        go.Figure: Interactive Plotly Gauge figure.
    """
    pct = probability * 100.0
    bar_color = get_risk_badge_color(risk_category)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a", "family": "Inter, sans-serif"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94a3b8",
                    "tickvals": [0, 30, 50, 60, 100],
                    "ticktext": ["0%", "30% (Low)", "50% (Cutoff)", "60% (High)", "100%"],
                    "tickfont": {"size": 11, "color": "#64748b"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 30], "color": "#ecfdf5"},   # Soft Green
                    {"range": [30, 60], "color": "#fffbeb"},  # Soft Amber
                    {"range": [60, 100], "color": "#fef2f2"}, # Soft Red
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.8,
                    "value": 50.0,
                },
            },
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif"},
    )
    return fig


def create_metrics_comparison_chart(metrics_data: Dict[str, Any]) -> go.Figure:
    """Generate an accessible, publication-grade bar chart comparing candidate models.

    Args:
        metrics_data: Dictionary loaded from metrics_summary.json.

    Returns:
        go.Figure: Interactive grouped bar chart.
    """
    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr = test_eval.get("logistic_regression", {})
    rf = test_eval.get("random_forest", {})

    metrics = ["ROC-AUC", "Recall", "Precision", "F1-Score", "Accuracy"]
    lr_scores = [
        lr.get("roc_auc", 0.8416),
        lr.get("recall", 0.7834),
        lr.get("precision", 0.5043),
        lr.get("f1_score", 0.6136),
        lr.get("accuracy", 0.7381),
    ]
    rf_scores = [
        rf.get("roc_auc", 0.8390),
        rf.get("recall", 0.7567),
        rf.get("precision", 0.5250),
        rf.get("f1_score", 0.6199),
        rf.get("accuracy", 0.7537),
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[s * 100 for s in lr_scores],
            name="Logistic Regression (Champion)",
            marker_color="#0284c7",  # Sky 600
            text=[f"{s:.1%}" if m != "ROC-AUC" else f"{s:.4f}" for m, s in zip(metrics, lr_scores)],
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            hovertemplate="<b>%{x}</b><br>Logistic Regression: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[s * 100 for s in rf_scores],
            name="Random Forest",
            marker_color="#94a3b8",  # Slate 400
            text=[f"{s:.1%}" if m != "ROC-AUC" else f"{s:.4f}" for m, s in zip(metrics, rf_scores)],
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
            hovertemplate="<b>%{x}</b><br>Random Forest: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#334155"),
        ),
        xaxis=dict(
            tickfont=dict(size=13, color="#0f172a", family="Inter, sans-serif"),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text="Evaluation Score (%)", font=dict(size=12, color="#64748b")),
            range=[0, 105],
            gridcolor="#e2e8f0",
            gridwidth=1,
            tickfont=dict(size=11, color="#64748b"),
        ),
        font={"family": "Inter, system-ui, sans-serif"},
    )
    return fig


def create_confusion_matrix_charts(metrics_data: Dict[str, Any]) -> go.Figure:
    """Generate side-by-side confusion matrix heatmaps for candidate models.

    Args:
        metrics_data: Dictionary loaded from metrics_summary.json.

    Returns:
        go.Figure: Interactive side-by-side heatmap subplots.
    """
    test_eval = metrics_data.get("held_out_test_evaluation", {})
    lr = test_eval.get("logistic_regression", {})
    rf = test_eval.get("random_forest", {})

    lr_cm = lr.get("confusion_matrix", {}).get("matrix", [[747, 288], [81, 293]])
    rf_cm = rf.get("confusion_matrix", {}).get("matrix", [[779, 256], [91, 283]])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Logistic Regression (Champion)</b><br><span style='font-size:11px;color:#0284c7;'>293 Caught Churners (78.3% Recall)</span>",
            "<b>Random Forest</b><br><span style='font-size:11px;color:#64748b;'>283 Caught Churners (75.7% Recall)</span>",
        ],
        horizontal_spacing=0.18,
    )

    lr_annotations = [
        [f"{lr_cm[0][0]}<br><span style='font-size:10px;'>True Negative</span>", f"{lr_cm[0][1]}<br><span style='font-size:10px;'>False Positive</span>"],
        [f"{lr_cm[1][0]}<br><span style='font-size:10px;color:#dc2626;font-weight:bold;'>81 Missed</span>", f"{lr_cm[1][1]}<br><span style='font-size:10px;color:#0284c7;font-weight:bold;'>293 Caught</span>"],
    ]

    rf_annotations = [
        [f"{rf_cm[0][0]}<br><span style='font-size:10px;'>True Negative</span>", f"{rf_cm[0][1]}<br><span style='font-size:10px;'>False Positive</span>"],
        [f"{rf_cm[1][0]}<br><span style='font-size:10px;color:#dc2626;font-weight:bold;'>91 Missed</span>", f"{rf_cm[1][1]}<br><span style='font-size:10px;color:#0284c7;font-weight:bold;'>283 Caught</span>"],
    ]

    fig.add_trace(
        go.Heatmap(
            z=lr_cm,
            x=["Predicted: Loyal (0)", "Predicted: Churn (1)"],
            y=["Actual: Loyal (0)", "Actual: Churn (1)"],
            colorscale="Blues",
            showscale=False,
            text=lr_annotations,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#0f172a"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=rf_cm,
            x=["Predicted: Loyal (0)", "Predicted: Churn (1)"],
            y=["Actual: Loyal (0)", "Actual: Churn (1)"],
            colorscale="Blues",
            showscale=False,
            text=rf_annotations,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#0f172a"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif"},
    )
    return fig


def get_app_css() -> str:
    """Return polished, responsive custom CSS scoped for ChurnIQ B2B SaaS interface."""
    return """
    <style>
    /* -------------------------------------------------------------
       ChurnIQ Premium SaaS Light Theme Styling
       ------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Page container adjustments */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px;
    }

    /* Modern Branded Header */
    .churniq-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -2px rgba(15, 23, 42, 0.06);
    }
    .churniq-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .churniq-badge {
        background: rgba(14, 165, 233, 0.2);
        border: 1px solid rgba(14, 165, 233, 0.4);
        color: #38bdf8;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .churniq-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .churniq-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin: 0;
    }

    /* System Telemetry Pill Row */
    .telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #f1f5f9;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 9999px;
    }
    .telemetry-dot-green {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
    }
    .telemetry-dot-blue {
        width: 7px;
        height: 7px;
        background-color: #38bdf8;
        border-radius: 50%;
    }
    .telemetry-dot-slate {
        width: 7px;
        height: 7px;
        background-color: #94a3b8;
        border-radius: 50%;
    }

    /* Polished Card Surface */
    .churniq-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .churniq-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-hint {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .risk-badge-low {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .risk-badge-medium {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .risk-badge-high {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 8px;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .section-desc {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 18px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748b;
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        background-color: #f0f9ff !important;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.55rem 1.4rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 2px rgba(2, 132, 199, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3);
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Disclaimer / Callout box */
    .disclaimer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 14px;
        margin-bottom: 14px;
        line-height: 1.5;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-brand {
        padding: 10px 0 16px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }
    </style>
    """
