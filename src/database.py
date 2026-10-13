"""Database integration module for Smart Customer Churn Prediction AI.

Manages secure MySQL connection pooling, schema initialization, parameterized prediction
logging, and retrieval of historical prediction metrics and logs.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector.cursor import MySQLCursor
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    MODEL_VERSION,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
    SCHEMA_SQL_PATH,
    get_db_config,
)
from src.predict import predict_single_customer


class DatabaseError(Exception):
    """Base exception for database operations."""


class DatabaseConfigError(DatabaseError):
    """Raised when database configuration or credentials are missing or invalid."""


class DatabaseConnectionError(DatabaseError):
    """Raised when establishing a connection to MySQL fails."""


class DatabaseQueryError(DatabaseError):
    """Raised when executing a database query fails."""


def validate_db_config(config: Dict[str, Any]) -> None:
    """Validate that required database configuration keys and values are present.

    Args:
        config: Dictionary containing MySQL connection parameters.

    Raises:
        DatabaseConfigError: If required keys are missing or credentials are blank.
    """
    if not isinstance(config, dict):
        raise DatabaseConfigError("Database configuration must be a dictionary.")

    required_keys = ["host", "port", "database", "user", "password"]
    missing = [k for k in required_keys if k not in config or config[k] is None]
    if missing:
        raise DatabaseConfigError(
            f"Missing required database configuration keys: {missing}. "
            "Please ensure environment variables (MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, "
            "MYSQL_USER, MYSQL_PASSWORD) are set."
        )

    # Validate that critical connection parameters are not empty strings
    empty_fields = [k for k in ["host", "database", "user"] if str(config.get(k, "")).strip() == ""]
    if empty_fields:
        raise DatabaseConfigError(
            f"Database configuration fields cannot be empty: {empty_fields}. "
            "Please set valid database connection details in your environment or .env file."
        )


def get_connection(
    config: Optional[Dict[str, Any]] = None,
    connect_timeout: int = 5,
) -> mysql.connector.MySQLConnection:
    """Establish and return an active MySQL connection.

    Args:
        config: Connection parameters (defaults to get_db_config()).
        connect_timeout: Connection timeout in seconds.

    Returns:
        mysql.connector.MySQLConnection: Active connection.

    Raises:
        DatabaseConfigError: If configuration is invalid.
        DatabaseConnectionError: If connection to MySQL fails.
    """
    cfg = config or get_db_config()
    validate_db_config(cfg)

    try:
        connection = mysql.connector.connect(
            host=cfg["host"],
            port=cfg["port"],
            database=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
            connect_timeout=connect_timeout,
            autocommit=False,
        )
        return connection
    except MySQLError as exc:
        # Strip password and internal traces from error message
        sanitized_msg = str(exc)
        if "password" in sanitized_msg.lower():
            sanitized_msg = "Access denied for user or invalid password."
        raise DatabaseConnectionError(
            f"Failed to connect to MySQL database '{cfg['database']}' at {cfg['host']}:{cfg['port']}. "
            f"Details: {sanitized_msg}"
        ) from exc


@contextmanager
def get_db_cursor(
    config: Optional[Dict[str, Any]] = None,
    dictionary: bool = True,
) -> Generator[Tuple[mysql.connector.MySQLConnection, MySQLCursor], None, None]:
    """Context manager for managing connection lifecycle, commits, and rollbacks.

    Automatically commits successful transactions and rolls back on failure.

    Yields:
        Tuple[MySQLConnection, MySQLCursor]: Connection and active cursor.
    """
    conn = get_connection(config)
    cursor = conn.cursor(dictionary=dictionary)
    try:
        yield conn, cursor
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        cursor.close()
        conn.close()


def test_connection(config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Verify database connectivity safely without throwing exceptions.

    Args:
        config: Optional database configuration override.

    Returns:
        Tuple[bool, str]: (True, 'Connection successful') or (False, error_message).
    """
    try:
        conn = get_connection(config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.fetchall()
        cursor.close()
        conn.close()
        return True, "Connection successful"
    except (DatabaseConfigError, DatabaseConnectionError) as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Unexpected connection error: {exc}"


def init_db(
    config: Optional[Dict[str, Any]] = None,
    schema_path: Union[Path, str] = SCHEMA_SQL_PATH,
) -> bool:
    """Initialize the prediction_logs table using the supplied SQL schema.

    Executes CREATE TABLE IF NOT EXISTS without dropping or modifying existing data.

    Args:
        config: Optional database configuration override.
        schema_path: Path to schema.sql file.

    Returns:
        bool: True if initialization succeeded.

    Raises:
        DatabaseQueryError: If table creation fails.
    """
    path = Path(schema_path)
    if not path.is_file():
        raise DatabaseQueryError(f"Database schema file not found at '{path}'.")

    sql_script = path.read_text(encoding="utf-8")

    try:
        with get_db_cursor(config, dictionary=False) as (_, cursor):
            # Split and execute non-empty statements
            statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
            for stmt in statements:
                cursor.execute(stmt)
        return True
    except MySQLError as exc:
        raise DatabaseQueryError(f"Database initialization failed: {exc}") from exc


def insert_prediction_log(
    prediction: Dict[str, Any],
    model_version: str = MODEL_VERSION,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a single prediction log record into the database using parameterized SQL.

    Args:
        prediction: Dictionary produced by predict_single_customer.
        model_version: Model release identifier.
        config: Optional database configuration override.

    Returns:
        int: The auto-generated primary key (id) of the inserted record.

    Raises:
        DatabaseQueryError: If validation or insertion fails.
    """
    # Validate required payload fields
    required_fields = ["model_name", "churn_probability", "predicted_churn", "risk_category"]
    missing = [f for f in required_fields if f not in prediction]
    if missing:
        raise DatabaseQueryError(f"Prediction data is missing required fields: {missing}")

    model_name = str(prediction["model_name"])
    prob = float(prediction["churn_probability"])
    predicted_churn = int(prediction["predicted_churn"])
    risk_category = str(prediction["risk_category"])
    threshold = float(prediction.get("decision_threshold", DEFAULT_THRESHOLD))

    # Validate value constraints
    if not (0.0 <= prob <= 1.0):
        raise DatabaseQueryError(f"Invalid churn_probability: {prob}. Must be between 0.0 and 1.0.")
    if predicted_churn not in (0, 1):
        raise DatabaseQueryError(f"Invalid predicted_churn: {predicted_churn}. Must be 0 or 1.")
    if risk_category not in (RISK_CATEGORY_LOW, RISK_CATEGORY_MEDIUM, RISK_CATEGORY_HIGH):
        raise DatabaseQueryError(
            f"Invalid risk_category: '{risk_category}'. "
            f"Must be one of: {RISK_CATEGORY_LOW}, {RISK_CATEGORY_MEDIUM}, {RISK_CATEGORY_HIGH}."
        )

    insert_sql = """
        INSERT INTO prediction_logs (
            model_name,
            model_version,
            churn_probability,
            predicted_churn,
            risk_category,
            decision_threshold
        ) VALUES (%s, %s, %s, %s, %s, %s);
    """
    params = (
        model_name,
        model_version,
        prob,
        predicted_churn,
        risk_category,
        threshold,
    )

    try:
        with get_db_cursor(config) as (_, cursor):
            cursor.execute(insert_sql, params)
            log_id = cursor.lastrowid
            return int(log_id)
    except MySQLError as exc:
        raise DatabaseQueryError(f"Failed to insert prediction log: {exc}") from exc


def get_recent_prediction_logs(
    limit: int = 50,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent prediction history logs using parameterized SQL.

    Args:
        limit: Maximum number of records to return.
        config: Optional database configuration override.

    Returns:
        List[Dict[str, Any]]: List of dictionary prediction logs ordered newest first.
    """
    safe_limit = max(1, min(int(limit), 1000))
    query = """
        SELECT
            id,
            created_at,
            model_name,
            model_version,
            churn_probability,
            predicted_churn,
            risk_category,
            decision_threshold
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT %s;
    """

    try:
        with get_db_cursor(config) as (_, cursor):
            cursor.execute(query, (safe_limit,))
            rows = cursor.fetchall()
            # Convert decimal/datetime fields to standard types
            results = []
            for row in rows:
                results.append(
                    {
                        "id": int(row["id"]),
                        "created_at": (
                            row["created_at"].isoformat()
                            if hasattr(row["created_at"], "isoformat")
                            else str(row["created_at"])
                        ),
                        "model_name": str(row["model_name"]),
                        "model_version": str(row["model_version"]),
                        "churn_probability": float(row["churn_probability"]),
                        "predicted_churn": int(row["predicted_churn"]),
                        "risk_category": str(row["risk_category"]),
                        "decision_threshold": float(row["decision_threshold"]),
                    }
                )
            return results
    except MySQLError as exc:
        raise DatabaseQueryError(f"Failed to fetch recent prediction logs: {exc}") from exc


def get_prediction_summary_stats(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate aggregate summary statistics across historical prediction logs.

    Returns:
        Dict[str, Any]: Total predictions, risk counts, churn counts, and mean probability.
    """
    query = """
        SELECT
            COUNT(*) AS total_predictions,
            COALESCE(SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END), 0) AS high_risk_predictions,
            COALESCE(SUM(CASE WHEN risk_category = 'Medium' THEN 1 ELSE 0 END), 0) AS medium_risk_predictions,
            COALESCE(SUM(CASE WHEN risk_category = 'Low' THEN 1 ELSE 0 END), 0) AS low_risk_predictions,
            COALESCE(SUM(CASE WHEN predicted_churn = 1 THEN 1 ELSE 0 END), 0) AS predicted_churn_count,
            COALESCE(SUM(CASE WHEN predicted_churn = 0 THEN 1 ELSE 0 END), 0) AS predicted_retention_count,
            COALESCE(AVG(churn_probability), 0.0) AS avg_churn_probability
        FROM prediction_logs;
    """

    try:
        with get_db_cursor(config) as (_, cursor):
            cursor.execute(query)
            row = cursor.fetchone()
            if not row or row["total_predictions"] == 0:
                return {
                    "total_predictions": 0,
                    "high_risk_predictions": 0,
                    "medium_risk_predictions": 0,
                    "low_risk_predictions": 0,
                    "predicted_churn_count": 0,
                    "predicted_retention_count": 0,
                    "avg_churn_probability": 0.0,
                }
            return {
                "total_predictions": int(row["total_predictions"]),
                "high_risk_predictions": int(row["high_risk_predictions"]),
                "medium_risk_predictions": int(row["medium_risk_predictions"]),
                "low_risk_predictions": int(row["low_risk_predictions"]),
                "predicted_churn_count": int(row["predicted_churn_count"]),
                "predicted_retention_count": int(row["predicted_retention_count"]),
                "avg_churn_probability": round(float(row["avg_churn_probability"]), 4),
            }
    except MySQLError as exc:
        raise DatabaseQueryError(f"Failed to compute prediction summary statistics: {exc}") from exc


def predict_and_log_customer(
    customer: Union[Dict[str, Any], pd.DataFrame],
    save_to_db: bool = True,
    model_version: str = MODEL_VERSION,
    db_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute inference for a customer and persist the result into MySQL when requested.

    Inference is strictly decoupled from database availability:
    - If save_to_db is False, only prediction inference runs.
    - If database insertion fails or MySQL is unavailable, the prediction result is
      still returned alongside an explicit logging failure status.
    - It never silently claims a record was saved if insertion did not succeed.

    Args:
        customer: Raw customer feature dictionary or DataFrame.
        save_to_db: Whether to attempt saving to the database.
        model_version: Model release identifier.
        db_config: Optional database configuration override.

    Returns:
        Dict[str, Any]: Combined prediction result and database logging status.
    """
    # 1. Run inference using pure model pipeline
    prediction_result = predict_single_customer(customer)
    result = dict(prediction_result)

    if not save_to_db:
        result["logged_to_db"] = False
        result["log_id"] = None
        return result

    # 2. Attempt database logging
    try:
        log_id = insert_prediction_log(prediction_result, model_version=model_version, config=db_config)
        result["logged_to_db"] = True
        result["log_id"] = log_id
        result["db_error"] = None
    except Exception as exc:
        result["logged_to_db"] = False
        result["log_id"] = None
        result["db_error"] = str(exc)

    return result
