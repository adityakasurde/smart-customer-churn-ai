"""Prediction engine and churn risk stratification module for Smart Customer Churn Prediction AI.

Loads the trained champion pipeline (Logistic Regression), validates raw customer input,
executes inference, and stratifies customers into Low, Medium, and High churn risk categories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CHAMPION_MODEL_NAME,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    ID_COLUMN,
    LOGISTIC_REGRESSION_MODEL_PATH,
    NUMERICAL_FEATURE_COLUMNS,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    TARGET_COLUMN,
)


class PredictionEngineError(Exception):
    """Base exception for all prediction engine errors."""


class ModelNotFoundError(PredictionEngineError):
    """Raised when the specified model artifact cannot be found on disk."""


class ModelLoadError(PredictionEngineError):
    """Raised when a model artifact exists but fails to load or is corrupted."""


class ValidationError(PredictionEngineError):
    """Raised when input customer data violates schema or business constraints."""


# Module-level model cache
_LOADED_MODEL_CACHE: Dict[str, Pipeline] = {}


def load_prediction_model(
    model_path: Union[Path, str] = LOGISTIC_REGRESSION_MODEL_PATH,
    force_reload: bool = False,
) -> Pipeline:
    """Safely load a trained Scikit-learn Pipeline artifact from disk.

    Args:
        model_path: Filepath to the .joblib artifact.
        force_reload: If True, bypasses the in-memory cache and reloads from disk.

    Returns:
        Pipeline: The loaded and verified Scikit-learn Pipeline.

    Raises:
        ModelNotFoundError: If the artifact does not exist at model_path.
        ModelLoadError: If the artifact is corrupted or not a valid Pipeline.
    """
    path = Path(model_path)
    cache_key = str(path.resolve())

    if not force_reload and cache_key in _LOADED_MODEL_CACHE:
        return _LOADED_MODEL_CACHE[cache_key]

    if not path.is_file():
        raise ModelNotFoundError(
            f"Trained model artifact not found at '{path}'. "
            "Please run 'python -m src.train' to generate model artifacts."
        )

    try:
        loaded_obj = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model artifact from '{path}'. The file may be corrupted."
        ) from exc

    if not isinstance(loaded_obj, Pipeline):
        raise ModelLoadError(
            f"Loaded object from '{path}' is of type {type(loaded_obj).__name__}, "
            "expected a scikit-learn Pipeline."
        )

    _LOADED_MODEL_CACHE[cache_key] = loaded_obj
    return loaded_obj


def categorize_churn_risk(
    probability: float,
    threshold_low: float = RISK_THRESHOLD_LOW,
    threshold_high: float = RISK_THRESHOLD_HIGH,
) -> str:
    """Stratify churn probability into business risk categories.

    Bands:
    - Low Risk: probability < threshold_low (default < 0.30)
    - Medium Risk: threshold_low <= probability < threshold_high (default 0.30 <= p < 0.60)
    - High Risk: probability >= threshold_high (default >= 0.60)

    Args:
        probability: Float between 0.0 and 1.0.
        threshold_low: Lower cutoff (default: 0.30).
        threshold_high: Upper cutoff (default: 0.60).

    Returns:
        str: 'Low', 'Medium', or 'High'

    Raises:
        ValueError: If probability is outside [0.0, 1.0].
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Churn probability must be between 0.0 and 1.0, got {probability}.")

    if probability < threshold_low:
        return RISK_CATEGORY_LOW
    elif probability < threshold_high:
        return RISK_CATEGORY_MEDIUM
    else:
        return RISK_CATEGORY_HIGH


def validate_customer_features(
    data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
) -> pd.DataFrame:
    """Validate customer feature dictionary or DataFrame against the training schema.

    Enforces:
    1. All 19 required predictive features exist.
    2. customerID is not required and safely stripped if present.
    3. Numeric fields are non-negative and valid numbers.
    4. SeniorCitizen is strictly 0 or 1.
    5. Categorical fields are non-empty strings.
    6. Blank TotalCharges with tenure == 0 is gracefully assigned 0.0.

    Args:
        data: Single customer dict, list of customer dicts, or DataFrame.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame matching FEATURE_COLUMNS order.

    Raises:
        ValidationError: If any validation rule is violated.
    """
    if data is None:
        raise ValidationError("Input customer data cannot be None.")

    # Standardize to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValidationError("Input customer list cannot be empty.")
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValidationError("Input customer DataFrame cannot be empty.")
        df = data.copy()
    else:
        raise ValidationError(
            f"Unsupported input type: {type(data).__name__}. Expected dict, list of dicts, or DataFrame."
        )

    # Ignore customerID or target column if present
    for col_to_ignore in [ID_COLUMN, TARGET_COLUMN]:
        if col_to_ignore in df.columns:
            df = df.drop(columns=[col_to_ignore])

    # Check for missing required features
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required feature(s): {missing_cols}. "
            f"Input must contain all {len(FEATURE_COLUMNS)} predictive attributes."
        )

    # Filter down to the exact expected feature columns
    df = df[FEATURE_COLUMNS].copy()

    # Pre-clean TotalCharges: if string and blank with tenure == 0, replace with "0.0" before casting
    for row_idx in range(len(df)):
        tenure_raw = df.at[row_idx, "tenure"]
        tc_raw = df.at[row_idx, "TotalCharges"]

        try:
            t_num = float(tenure_raw)
        except (ValueError, TypeError):
            t_num = None

        if isinstance(tc_raw, str) and tc_raw.strip() == "":
            if t_num == 0:
                df.at[row_idx, "TotalCharges"] = "0.0"
            else:
                raise ValidationError(
                    f"Row {row_idx}: Field 'TotalCharges' cannot be empty when tenure is greater than 0."
                )

    # Validate and cast numeric columns to float/int
    for num_col in NUMERICAL_FEATURE_COLUMNS:
        raw_series = df[num_col]
        converted = pd.to_numeric(
            raw_series.astype(str).str.strip()
            if raw_series.dtype == object or getattr(raw_series.dtype, "name", "") in ("string", "str")
            else raw_series,
            errors="coerce",
        )
        for row_idx, val in converted.items():
            raw_val = df.at[row_idx, num_col]
            if pd.isna(val):
                raise ValidationError(
                    f"Row {row_idx}: Field '{num_col}' must be a valid number. Got: {raw_val!r}"
                )
            if val < 0:
                raise ValidationError(
                    f"Row {row_idx}: Field '{num_col}' must be a non-negative number. Got: {raw_val!r}"
                )
            if num_col == "SeniorCitizen" and val not in (0, 1):
                raise ValidationError(
                    f"Row {row_idx}: Field 'SeniorCitizen' must be 0 or 1. Got: {raw_val!r}"
                )
        df[num_col] = converted

    # Validate categorical fields (must be non-empty strings)
    cat_cols = [col for col in FEATURE_COLUMNS if col not in NUMERICAL_FEATURE_COLUMNS]
    for cat_col in cat_cols:
        for row_idx, val in df[cat_col].items():
            if val is None or pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                raise ValidationError(
                    f"Row {row_idx}: Categorical field '{cat_col}' cannot be null or empty."
                )
            df.at[row_idx, cat_col] = str(val)

    return df



def predict_single_customer(
    customer: Union[Dict[str, Any], pd.DataFrame],
    model_path: Union[Path, str] = LOGISTIC_REGRESSION_MODEL_PATH,
) -> Dict[str, Any]:
    """Predict churn probability and risk category for a single customer record.

    Uses the full Scikit-learn Pipeline (preprocessing + Logistic Regression)
    loaded from disk.

    Args:
        customer: Dictionary of raw customer attributes or 1-row DataFrame.
        model_path: Path to serialized model Pipeline.

    Returns:
        Dict[str, Any]:
            - churn_probability: float (0.0 to 1.0, 4 decimal places)
            - churn_probability_percent: float (0.0 to 100.0)
            - predicted_churn: int (0 or 1 using 0.5 threshold)
            - risk_category: str ('Low', 'Medium', or 'High')
            - model_name: str ('Logistic Regression')
            - decision_threshold: float (0.5)
    """
    validated_df = validate_customer_features(customer)
    if len(validated_df) != 1:
        raise ValidationError(
            f"predict_single_customer expects 1 record, got {len(validated_df)}."
        )

    pipeline = load_prediction_model(model_path)
    probas = pipeline.predict_proba(validated_df)[0]
    proba_churn = float(probas[1])

    risk = categorize_churn_risk(proba_churn)
    pred_class = 1 if proba_churn >= DEFAULT_THRESHOLD else 0

    return {
        "churn_probability": round(proba_churn, 4),
        "churn_probability_percent": round(proba_churn * 100, 2),
        "predicted_churn": int(pred_class),
        "risk_category": risk,
        "model_name": CHAMPION_MODEL_NAME,
        "decision_threshold": float(DEFAULT_THRESHOLD),
    }


def predict_batch(
    customers: Union[List[Dict[str, Any]], pd.DataFrame],
    model_path: Union[Path, str] = LOGISTIC_REGRESSION_MODEL_PATH,
) -> List[Dict[str, Any]]:
    """Predict churn probability and risk category for a batch of customer records.

    Args:
        customers: List of customer dictionaries or DataFrame.
        model_path: Path to serialized model Pipeline.

    Returns:
        List[Dict[str, Any]]: List of prediction result dictionaries.
    """
    validated_df = validate_customer_features(customers)
    pipeline = load_prediction_model(model_path)

    probas = pipeline.predict_proba(validated_df)
    results: List[Dict[str, Any]] = []

    for idx, proba_pair in enumerate(probas):
        proba_churn = float(proba_pair[1])
        risk = categorize_churn_risk(proba_churn)
        pred_class = 1 if proba_churn >= DEFAULT_THRESHOLD else 0

        results.append(
            {
                "churn_probability": round(proba_churn, 4),
                "churn_probability_percent": round(proba_churn * 100, 2),
                "predicted_churn": int(pred_class),
                "risk_category": risk,
                "model_name": CHAMPION_MODEL_NAME,
                "decision_threshold": float(DEFAULT_THRESHOLD),
            }
        )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTION ENGINE & RISK STRATIFICATION TEST")
    print("=" * 60)

    # Sample customer record
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

    result = predict_single_customer(sample_customer)
    print(f"Sample Customer Prediction Result:")
    for key, value in result.items():
        print(f"  {key:<28}: {value}")
