"""Unit tests for the prediction engine, schema validation, and churn risk stratification."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_THRESHOLD,
    LOGISTIC_REGRESSION_MODEL_PATH,
    RISK_CATEGORY_HIGH,
    RISK_CATEGORY_LOW,
    RISK_CATEGORY_MEDIUM,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
)
from src.predict import (
    ModelLoadError,
    ModelNotFoundError,
    ValidationError,
    categorize_churn_risk,
    load_prediction_model,
    predict_batch,
    predict_single_customer,
    validate_customer_features,
)


@pytest.fixture
def valid_customer_dict() -> dict:
    """Return a valid representative raw customer record."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
    }


def test_load_prediction_model_success():
    """Verify that the champion model artifact loads as a valid Pipeline."""
    model = load_prediction_model(LOGISTIC_REGRESSION_MODEL_PATH, force_reload=True)
    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_load_prediction_model_missing(tmp_path: Path):
    """Verify ModelNotFoundError is raised with clear message when file is missing."""
    missing_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(ModelNotFoundError, match="Trained model artifact not found"):
        load_prediction_model(missing_path, force_reload=True)


def test_load_prediction_model_corrupted(tmp_path: Path):
    """Verify ModelLoadError is raised when model file contains invalid/corrupted bytes."""
    corrupt_path = tmp_path / "corrupted_model.joblib"
    corrupt_path.write_bytes(b"THIS_IS_CORRUPT_DATA_NOT_A_JOBLIB_FILE")
    with pytest.raises(ModelLoadError, match="Failed to load model artifact"):
        load_prediction_model(corrupt_path, force_reload=True)


def test_load_prediction_model_wrong_object_type(tmp_path: Path):
    """Verify ModelLoadError is raised when artifact is not a Scikit-learn Pipeline."""
    import joblib

    fake_path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"key": "value"}, fake_path)
    with pytest.raises(ModelLoadError, match="expected a scikit-learn Pipeline"):
        load_prediction_model(fake_path, force_reload=True)


def test_predict_single_customer(valid_customer_dict: dict):
    """Verify single customer prediction produces expected structure, types, and values."""
    res = predict_single_customer(valid_customer_dict)

    assert isinstance(res, dict)
    assert "churn_probability" in res
    assert "churn_probability_percent" in res
    assert "predicted_churn" in res
    assert "risk_category" in res
    assert "model_name" in res
    assert "decision_threshold" in res

    assert 0.0 <= res["churn_probability"] <= 1.0
    assert 0.0 <= res["churn_probability_percent"] <= 100.0
    assert res["predicted_churn"] in (0, 1)
    assert res["risk_category"] in (RISK_CATEGORY_LOW, RISK_CATEGORY_MEDIUM, RISK_CATEGORY_HIGH)
    assert res["decision_threshold"] == DEFAULT_THRESHOLD

    # Verify probability percentage matches decimal probability
    assert pytest.approx(res["churn_probability"] * 100, abs=0.1) == res["churn_probability_percent"]

    # Verify predicted_churn matches 0.5 decision threshold
    expected_class = 1 if res["churn_probability"] >= DEFAULT_THRESHOLD else 0
    assert res["predicted_churn"] == expected_class


def test_predict_batch(valid_customer_dict: dict):
    """Verify batch predictions handle multiple customer records."""
    customer_2 = valid_customer_dict.copy()
    customer_2["tenure"] = 60
    customer_2["Contract"] = "Two year"
    customer_2["MonthlyCharges"] = 25.0
    customer_2["TotalCharges"] = 1500.0

    batch = [valid_customer_dict, customer_2]
    results = predict_batch(batch)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["churn_probability"] <= 1.0
        assert r["predicted_churn"] in (0, 1)


def test_categorize_churn_risk_boundaries():
    """Verify risk categorization strictly obeys documented business cutoffs and boundaries."""
    # Low Risk: probability < 0.30
    assert categorize_churn_risk(0.0) == RISK_CATEGORY_LOW
    assert categorize_churn_risk(0.15) == RISK_CATEGORY_LOW
    assert categorize_churn_risk(0.2999) == RISK_CATEGORY_LOW

    # Boundary at 0.30 -> Medium Risk
    assert categorize_churn_risk(RISK_THRESHOLD_LOW) == RISK_CATEGORY_MEDIUM
    assert categorize_churn_risk(0.45) == RISK_CATEGORY_MEDIUM
    assert categorize_churn_risk(0.5999) == RISK_CATEGORY_MEDIUM

    # Boundary at 0.60 -> High Risk
    assert categorize_churn_risk(RISK_THRESHOLD_HIGH) == RISK_CATEGORY_HIGH
    assert categorize_churn_risk(0.75) == RISK_CATEGORY_HIGH
    assert categorize_churn_risk(1.0) == RISK_CATEGORY_HIGH

    # Out of range values must raise ValueError
    with pytest.raises(ValueError, match="Churn probability must be between 0.0 and 1.0"):
        categorize_churn_risk(-0.01)
    with pytest.raises(ValueError, match="Churn probability must be between 0.0 and 1.0"):
        categorize_churn_risk(1.01)


def test_validation_missing_required_fields(valid_customer_dict: dict):
    """Verify ValidationError is raised when essential features are missing."""
    incomplete = valid_customer_dict.copy()
    del incomplete["Contract"]
    with pytest.raises(ValidationError, match="Missing required feature"):
        validate_customer_features(incomplete)


def test_validation_extra_fields_and_customerid_ignored(valid_customer_dict: dict):
    """Verify customerID and non-model metadata fields are safely stripped and ignored."""
    with_extras = valid_customer_dict.copy()
    with_extras["customerID"] = "TEST-1234-CUST"
    with_extras["notes"] = "VIP priority customer"
    with_extras["internal_crm_id"] = 99482

    validated_df = validate_customer_features(with_extras)
    assert "customerID" not in validated_df.columns
    assert "notes" not in validated_df.columns
    assert len(validated_df.columns) == 19

    # Should predict successfully without needing customerID as a feature
    res = predict_single_customer(with_extras)
    assert res["predicted_churn"] in (0, 1)


def test_validation_negative_tenure(valid_customer_dict: dict):
    """Verify ValidationError is raised for negative tenure."""
    invalid = valid_customer_dict.copy()
    invalid["tenure"] = -5
    with pytest.raises(ValidationError, match="Field 'tenure' must be a non-negative number"):
        validate_customer_features(invalid)


def test_validation_negative_charges(valid_customer_dict: dict):
    """Verify ValidationError is raised for negative MonthlyCharges or TotalCharges."""
    invalid_mc = valid_customer_dict.copy()
    invalid_mc["MonthlyCharges"] = -25.5
    with pytest.raises(ValidationError, match="Field 'MonthlyCharges' must be a non-negative number"):
        validate_customer_features(invalid_mc)

    invalid_tc = valid_customer_dict.copy()
    invalid_tc["TotalCharges"] = -100.0
    with pytest.raises(ValidationError, match="Field 'TotalCharges' must be a non-negative number"):
        validate_customer_features(invalid_tc)


def test_validation_invalid_senior_citizen(valid_customer_dict: dict):
    """Verify ValidationError is raised when SeniorCitizen is not 0 or 1."""
    invalid = valid_customer_dict.copy()
    invalid["SeniorCitizen"] = 3
    with pytest.raises(ValidationError, match="Field 'SeniorCitizen' must be 0 or 1"):
        validate_customer_features(invalid)


def test_validation_empty_categorical_field(valid_customer_dict: dict):
    """Verify ValidationError is raised when categorical fields are empty or null."""
    invalid = valid_customer_dict.copy()
    invalid["gender"] = "   "
    with pytest.raises(ValidationError, match="Categorical field 'gender' cannot be null or empty"):
        validate_customer_features(invalid)


def test_validation_blank_totalcharges_zero_tenure(valid_customer_dict: dict):
    """Verify TotalCharges blank string is converted to 0.0 when tenure is 0."""
    new_user = valid_customer_dict.copy()
    new_user["tenure"] = 0
    new_user["TotalCharges"] = " "
    validated = validate_customer_features(new_user)
    assert validated["TotalCharges"].iloc[0] == 0.0


def test_unseen_categorical_value_handling(valid_customer_dict: dict):
    """Verify that unseen categorical values do not crash the pipeline and yield valid predictions."""
    unseen_user = valid_customer_dict.copy()
    unseen_user["Contract"] = "Decade-to-Decade"  # Unseen category
    unseen_user["PaymentMethod"] = "Cryptocurrency"  # Unseen category

    res = predict_single_customer(unseen_user)
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["predicted_churn"] in (0, 1)
    assert res["risk_category"] in (RISK_CATEGORY_LOW, RISK_CATEGORY_MEDIUM, RISK_CATEGORY_HIGH)


def test_json_serializability(valid_customer_dict: dict):
    """Ensure prediction output dictionary is strictly JSON serializable."""
    res = predict_single_customer(valid_customer_dict)
    json_str = json.dumps(res)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["decision_threshold"] == 0.5
