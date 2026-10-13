"""Configuration constants, file paths, and feature definitions for Smart Customer Churn Prediction AI."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

# Raw Dataset
RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# Attribution
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DATASET_ATTRIBUTION = "IBM Business Analytics Telco Customer Churn Dataset"

# Special Columns
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

# Expected Schema Columns
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# Predictive Features (Excludes customerID and Churn)
FEATURE_COLUMNS = [col for col in EXPECTED_COLUMNS if col not in (ID_COLUMN, TARGET_COLUMN)]

NUMERICAL_FEATURE_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURE_COLUMNS = [
    col for col in FEATURE_COLUMNS if col not in NUMERICAL_FEATURE_COLUMNS
]

# Train / Test Splitting Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFY = True

# Evaluation & Modeling Parameters
DEFAULT_THRESHOLD = 0.5
CV_FOLDS = 5
POSITIVE_CLASS_LABEL = 1
CHAMPION_MODEL_NAME = "Logistic Regression"

# Churn Risk Categorization (Business Rules)
# - Low Risk: probability < 0.30
# - Medium Risk: 0.30 <= probability < 0.60
# - High Risk: probability >= 0.60
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_HIGH = 0.60

RISK_CATEGORY_LOW = "Low"
RISK_CATEGORY_MEDIUM = "Medium"
RISK_CATEGORY_HIGH = "High"

# Database & Prediction History
SCHEMA_SQL_PATH = SQL_DIR / "schema.sql"
MODEL_VERSION = "1.0.0"

# Artifact Paths
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
METRICS_SUMMARY_PATH = MODELS_DIR / "metrics_summary.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def get_db_config() -> dict:
    """Retrieve MySQL database configuration from environment variables."""
    import os
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    port_raw = os.getenv("MYSQL_PORT", "3306")
    try:
        port = int(port_raw)
    except ValueError:
        port = 3306

    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": port,
        "database": os.getenv("MYSQL_DATABASE", "churn_db"),
        "user": os.getenv("MYSQL_USER", ""),
        "password": os.getenv("MYSQL_PASSWORD", ""),
    }



