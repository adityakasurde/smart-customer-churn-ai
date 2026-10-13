"""Reusable UI helpers, metrics loaders, and form specifications for Streamlit dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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
