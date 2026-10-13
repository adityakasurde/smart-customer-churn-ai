"""Unit and integration tests for the MySQL database integration module."""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.database import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseQueryError,
    get_connection,
    get_prediction_summary_stats,
    get_recent_prediction_logs,
    init_db,
    insert_prediction_log,
    predict_and_log_customer,
    test_connection as db_test_connection,
    validate_db_config,
)



@pytest.fixture
def valid_db_config() -> dict:
    """Return a mock valid database configuration."""
    return {
        "host": "localhost",
        "port": 3306,
        "database": "test_churn_db",
        "user": "test_user",
        "password": "test_password",
    }


@pytest.fixture
def sample_prediction_data() -> dict:
    """Return a valid prediction result dictionary produced by the prediction engine."""
    return {
        "churn_probability": 0.7550,
        "churn_probability_percent": 75.50,
        "predicted_churn": 1,
        "risk_category": "High",
        "model_name": "Logistic Regression",
        "decision_threshold": 0.5,
    }


def test_validate_db_config_success(valid_db_config: dict):
    """Verify that a complete configuration dictionary passes validation."""
    validate_db_config(valid_db_config)  # Should not raise


def test_validate_db_config_missing_keys():
    """Verify DatabaseConfigError is raised when required keys are missing."""
    incomplete_cfg = {"host": "localhost", "port": 3306}
    with pytest.raises(DatabaseConfigError, match="Missing required database configuration"):
        validate_db_config(incomplete_cfg)


def test_validate_db_config_empty_fields(valid_db_config: dict):
    """Verify DatabaseConfigError is raised when critical fields are empty strings."""
    empty_cfg = valid_db_config.copy()
    empty_cfg["user"] = "   "
    with pytest.raises(DatabaseConfigError, match="cannot be empty"):
        validate_db_config(empty_cfg)


def test_validate_db_config_not_a_dict():
    """Verify DatabaseConfigError is raised when config is not a dictionary."""
    with pytest.raises(DatabaseConfigError, match="must be a dictionary"):
        validate_db_config("not_a_dict")


@patch("src.database.mysql.connector.connect")
def test_get_connection_success(mock_connect, valid_db_config: dict):
    """Verify get_connection establishes connection with expected parameters."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    conn = get_connection(valid_db_config)
    assert conn == mock_conn
    mock_connect.assert_called_once_with(
        host="localhost",
        port=3306,
        database="test_churn_db",
        user="test_user",
        password="test_password",
        connect_timeout=5,
        autocommit=False,
    )


@patch("src.database.mysql.connector.connect")
def test_get_connection_failure(mock_connect, valid_db_config: dict):
    """Verify DatabaseConnectionError is raised without leaking passwords."""
    from mysql.connector import Error as MySQLError

    mock_connect.side_effect = MySQLError("Access denied for user 'test_user'@'localhost' (using password: YES)")

    with pytest.raises(DatabaseConnectionError) as exc_info:
        get_connection(valid_db_config)

    error_msg = str(exc_info.value)
    # Password value itself must never appear in error message
    assert "test_password" not in error_msg
    assert "Failed to connect to MySQL database" in error_msg


@patch("src.database.get_connection")
def test_test_connection_utility(mock_get_conn, valid_db_config: dict):
    """Verify test_connection returns True on success and False with message on error."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_conn.return_value = mock_conn

    success, msg = db_test_connection(valid_db_config)
    assert success is True
    assert msg == "Connection successful"
    mock_cursor.execute.assert_called_once_with("SELECT 1;")

    # Now simulate failure
    mock_get_conn.side_effect = DatabaseConnectionError("Mock network failure")
    success, msg = db_test_connection(valid_db_config)
    assert success is False
    assert "Mock network failure" in msg



@patch("src.database.get_db_cursor")
def test_insert_prediction_log_parameterized(mock_get_cursor, sample_prediction_data: dict, valid_db_config: dict):
    """Verify insert_prediction_log executes a parameterized INSERT query and returns record ID."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42
    mock_get_cursor.return_value.__enter__.return_value = (mock_conn, mock_cursor)

    log_id = insert_prediction_log(sample_prediction_data, model_version="1.0.0", config=valid_db_config)
    assert log_id == 42

    # Verify query used parameterized %s placeholders
    mock_cursor.execute.assert_called_once()
    called_sql, called_params = mock_cursor.execute.call_args[0]
    assert "INSERT INTO prediction_logs" in called_sql
    assert "%s" in called_sql
    assert called_params == (
        "Logistic Regression",
        "1.0.0",
        0.7550,
        1,
        "High",
        0.5,
    )


def test_insert_prediction_log_invalid_values(sample_prediction_data: dict, valid_db_config: dict):
    """Verify DatabaseQueryError is raised for invalid probabilities or risk tiers."""
    # Invalid probability > 1.0
    invalid_prob = sample_prediction_data.copy()
    invalid_prob["churn_probability"] = 1.5
    with pytest.raises(DatabaseQueryError, match="Invalid churn_probability"):
        insert_prediction_log(invalid_prob, config=valid_db_config)

    # Invalid risk category
    invalid_risk = sample_prediction_data.copy()
    invalid_risk["risk_category"] = "CriticalExtreme"
    with pytest.raises(DatabaseQueryError, match="Invalid risk_category"):
        insert_prediction_log(invalid_risk, config=valid_db_config)

    # Invalid predicted churn (e.g. 2)
    invalid_churn = sample_prediction_data.copy()
    invalid_churn["predicted_churn"] = 2
    with pytest.raises(DatabaseQueryError, match="Invalid predicted_churn"):
        insert_prediction_log(invalid_churn, config=valid_db_config)


@patch("src.database.get_db_cursor")
def test_get_recent_prediction_logs(mock_get_cursor, valid_db_config: dict):
    """Verify get_recent_prediction_logs retrieves records with parameterized LIMIT."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_rows = [
        {
            "id": 1,
            "created_at": "2026-10-10T22:00:00",
            "model_name": "Logistic Regression",
            "model_version": "1.0.0",
            "churn_probability": 0.85,
            "predicted_churn": 1,
            "risk_category": "High",
            "decision_threshold": 0.5,
        }
    ]
    mock_cursor.fetchall.return_value = mock_rows
    mock_get_cursor.return_value.__enter__.return_value = (mock_conn, mock_cursor)

    logs = get_recent_prediction_logs(limit=10, config=valid_db_config)
    assert len(logs) == 1
    assert logs[0]["id"] == 1
    assert logs[0]["risk_category"] == "High"

    called_sql, called_params = mock_cursor.execute.call_args[0]
    assert "LIMIT %s" in called_sql
    assert called_params == (10,)


@patch("src.database.get_db_cursor")
def test_get_prediction_summary_stats(mock_get_cursor, valid_db_config: dict):
    """Verify get_prediction_summary_stats correctly aggregates historical metrics."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "total_predictions": 100,
        "high_risk_predictions": 30,
        "medium_risk_predictions": 25,
        "low_risk_predictions": 45,
        "predicted_churn_count": 35,
        "predicted_retention_count": 65,
        "avg_churn_probability": 0.4215,
    }
    mock_get_cursor.return_value.__enter__.return_value = (mock_conn, mock_cursor)

    stats = get_prediction_summary_stats(config=valid_db_config)
    assert stats["total_predictions"] == 100
    assert stats["high_risk_predictions"] == 30
    assert stats["avg_churn_probability"] == 0.4215


@patch("src.database.get_connection")
def test_transaction_rollback_on_failure(mock_get_conn, sample_prediction_data: dict, valid_db_config: dict):
    """Verify that failed transactions trigger connection rollback."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("Simulated database disk failure")
    mock_conn.cursor.return_value = mock_cursor
    mock_get_conn.return_value = mock_conn

    with pytest.raises(Exception, match="Simulated database disk failure"):
        insert_prediction_log(sample_prediction_data, config=valid_db_config)

    # Rollback must be called when an exception occurs inside get_db_cursor
    mock_conn.rollback.assert_called_once()


def test_predict_and_log_graceful_degradation_without_db():
    """Verify predict_and_log_customer completes inference even when MySQL credentials are missing."""
    sample_customer = {
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
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00,
    }

    # Calling with empty credentials (no DB configured)
    empty_db_config = {"host": "localhost", "port": 3306, "database": "churn_db", "user": "", "password": ""}
    result = predict_and_log_customer(sample_customer, save_to_db=True, db_config=empty_db_config)

    # Pure model inference MUST succeed
    assert "churn_probability" in result
    assert result["churn_probability"] > 0.80
    assert result["risk_category"] == "High"
    assert result["predicted_churn"] == 1

    # Database logging status MUST indicate failure cleanly without claiming success
    assert result["logged_to_db"] is False
    assert result["log_id"] is None
    assert result["db_error"] is not None
    assert "cannot be empty" in result["db_error"]


@patch("src.database.insert_prediction_log")
def test_predict_and_log_success(mock_insert):
    """Verify predict_and_log_customer attaches log_id when save succeeds."""
    mock_insert.return_value = 99

    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 50,
        "PhoneService": "Yes",
        "MultipleLines": "No",
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
        "MonthlyCharges": 40.0,
        "TotalCharges": 2000.0,
    }

    result = predict_and_log_customer(sample_customer, save_to_db=True)
    assert result["logged_to_db"] is True
    assert result["log_id"] == 99
    assert result["db_error"] is None


@pytest.mark.skipif(
    not os.getenv("RUN_MYSQL_INTEGRATION"),
    reason="Live MySQL integration tests disabled by default. Set RUN_MYSQL_INTEGRATION=1 to enable.",
)
def test_live_mysql_integration(valid_db_config: dict):
    """Opt-in live integration test against a dedicated local MySQL test database."""
    # Ensure connectivity
    conn_ok, conn_msg = test_connection(valid_db_config)
    if not conn_ok:
        pytest.skip(f"Live MySQL test instance not reachable: {conn_msg}")

    # Initialize schema
    assert init_db(valid_db_config) is True

    # Insert test prediction
    test_pred = {
        "churn_probability": 0.4200,
        "predicted_churn": 0,
        "risk_category": "Medium",
        "model_name": "Logistic Regression",
        "decision_threshold": 0.5,
    }
    inserted_id = insert_prediction_log(test_pred, config=valid_db_config)
    assert inserted_id > 0

    # Retrieve recent logs
    recent = get_recent_prediction_logs(limit=5, config=valid_db_config)
    assert any(r["id"] == inserted_id for r in recent)
