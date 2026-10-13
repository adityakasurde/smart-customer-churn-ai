"""Unit tests for the Streamlit dashboard helpers and UI logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.ui_components import (
    DEFAULT_CUSTOMER_FORM_VALUES,
    FEATURE_OPTIONS,
    get_factual_interpretation,
    get_risk_badge_color,
    load_model_metadata,
    load_model_metrics,
)
from src.config import (
    METRICS_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
)
from src.database import predict_and_log_customer
from src.predict import predict_single_customer


def test_load_model_metrics_success():
    """Verify that existing metrics_summary.json loads successfully and contains required keys."""
    metrics, err = load_model_metrics(METRICS_SUMMARY_PATH)
    assert err is None
    assert isinstance(metrics, dict)
    assert "held_out_test_evaluation" in metrics
    assert "logistic_regression" in metrics["held_out_test_evaluation"]
    assert "random_forest" in metrics["held_out_test_evaluation"]


def test_load_model_metrics_missing(tmp_path: Path):
    """Verify that load_model_metrics returns a graceful error message when the file is absent."""
    missing_file = tmp_path / "non_existent_metrics.json"
    metrics, err = load_model_metrics(missing_file)
    assert metrics is None
    assert err is not None
    assert "not found" in err


def test_load_model_metrics_malformed(tmp_path: Path):
    """Verify that load_model_metrics handles corrupt/invalid JSON without crashing."""
    corrupt_file = tmp_path / "corrupt_metrics.json"
    corrupt_file.write_text("NOT_VALID_JSON{:::}", encoding="utf-8")
    metrics, err = load_model_metrics(corrupt_file)
    assert metrics is None
    assert err is not None
    assert "Failed to parse" in err


def test_load_model_metadata_success():
    """Verify that model_metadata.json loads successfully."""
    meta, err = load_model_metadata(MODEL_METADATA_PATH)
    assert err is None
    assert isinstance(meta, dict)
    assert "selected_champion" in meta
    assert meta["selected_champion"] == "logistic_regression"


def test_get_factual_interpretation():
    """Verify factual interpretation statements for each risk category without overclaims."""
    high_text = get_factual_interpretation(0.85, RISK_CATEGORY_HIGH)
    assert "High Risk tier" in high_text
    assert "85.0% churn probability" in high_text
    assert "not a deterministic guarantee" in high_text

    med_text = get_factual_interpretation(0.45, RISK_CATEGORY_MEDIUM)
    assert "Medium Risk tier" in med_text
    assert "45.0% churn probability" in med_text

    low_text = get_factual_interpretation(0.10, RISK_CATEGORY_LOW)
    assert "Low Risk tier" in low_text
    assert "10.0% churn probability" in low_text


def test_get_risk_badge_color():
    """Verify risk badge colors map consistently."""
    assert get_risk_badge_color(RISK_CATEGORY_HIGH) == "#e63946"
    assert get_risk_badge_color(RISK_CATEGORY_MEDIUM) == "#f4a261"
    assert get_risk_badge_color(RISK_CATEGORY_LOW) == "#2a9d8f"


def test_default_customer_form_values_schema():
    """Verify that default form values cover all 19 required features with valid types."""
    from src.config import FEATURE_COLUMNS

    for col in FEATURE_COLUMNS:
        assert col in DEFAULT_CUSTOMER_FORM_VALUES, f"Missing default value for feature: {col}"

    assert len(DEFAULT_CUSTOMER_FORM_VALUES) == 19
    assert DEFAULT_CUSTOMER_FORM_VALUES["tenure"] >= 0
    assert DEFAULT_CUSTOMER_FORM_VALUES["MonthlyCharges"] >= 0.0
    assert DEFAULT_CUSTOMER_FORM_VALUES["TotalCharges"] >= 0.0


def test_dashboard_prediction_flow_integration():
    """Verify that calling predict_single_customer with default form values produces a valid result."""
    res = predict_single_customer(DEFAULT_CUSTOMER_FORM_VALUES)
    assert "churn_probability" in res
    assert "risk_category" in res
    assert 0.0 <= res["churn_probability"] <= 1.0


def test_dashboard_database_unavailable_handling():
    """Verify predict_and_log_customer operates cleanly when DB credentials are intentionally invalid."""
    invalid_db_config = {
        "host": "localhost",
        "port": 3306,
        "database": "non_existent_db",
        "user": "",
        "password": "",
    }
    result = predict_and_log_customer(
        DEFAULT_CUSTOMER_FORM_VALUES, save_to_db=True, db_config=invalid_db_config
    )

    # Prediction must succeed
    assert "churn_probability" in result
    assert result["churn_probability"] > 0
    # Database status must indicate failure cleanly without crashing
    assert result["logged_to_db"] is False
    assert result["log_id"] is None
    assert result["db_error"] is not None
