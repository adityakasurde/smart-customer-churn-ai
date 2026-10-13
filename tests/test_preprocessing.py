"""Automated unit tests for data loading and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data_loader import validate_raw_data
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    create_train_test_split,
    get_feature_types,
    separate_features_target,
)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Provide a minimal synthetic raw DataFrame mirroring Telco schema."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004", "C005", "C006"],
            "gender": ["Female", "Male", "Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
            "tenure": [1, 0, 24, 0, 12, 60],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
            "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "No internet service", "Yes", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No", "No", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No", "Yes", "No"],
            "StreamingMovies": ["No", "No", "No internet service", "Yes", "Yes", "No"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 20.25, 45.0, 60.0, 85.5, 30.0],
            "TotalCharges": ["29.85", " ", "1080.0", "", "1026.0", "1800.0"],
            "Churn": ["No", "No", "No", "Yes", "Yes", "No"],
        }
    )


def test_validate_raw_data_success(synthetic_raw_df: pd.DataFrame):
    """Test validation passes with complete schema."""
    assert validate_raw_data(synthetic_raw_df) is True


def test_validate_raw_data_missing_column(synthetic_raw_df: pd.DataFrame):
    """Test validation raises ValueError when required column is missing."""
    df_missing = synthetic_raw_df.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_raw_data(df_missing)


def test_clean_data_totalcharges_conversion_and_zero_tenure(synthetic_raw_df: pd.DataFrame):
    """Test TotalCharges blank strings are converted and zero-tenure rows become 0.0."""
    cleaned = clean_data(synthetic_raw_df)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    # C002 and C004 had tenure == 0 with blank TotalCharges -> should be 0.0
    c002_val = cleaned.loc[cleaned["customerID"] == "C002", "TotalCharges"].values[0]
    c004_val = cleaned.loc[cleaned["customerID"] == "C004", "TotalCharges"].values[0]
    assert c002_val == 0.0
    assert c004_val == 0.0

    # Non-zero tenure should convert to float accurately
    c003_val = cleaned.loc[cleaned["customerID"] == "C003", "TotalCharges"].values[0]
    assert c003_val == 1080.0


def test_clean_data_preserves_original_df(synthetic_raw_df: pd.DataFrame):
    """Test that clean_data does not mutate the raw input DataFrame."""
    original_tc = synthetic_raw_df["TotalCharges"].copy()
    _ = clean_data(synthetic_raw_df)
    pd.testing.assert_series_equal(synthetic_raw_df["TotalCharges"], original_tc)


def test_clean_data_target_encoding(synthetic_raw_df: pd.DataFrame):
    """Test that Churn is cleanly converted from Yes/No into 1/0 integers."""
    cleaned = clean_data(synthetic_raw_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned["Churn"].dtype == int

    # C004 and C005 were 'Yes' -> 1; others were 'No' -> 0
    assert cleaned.loc[cleaned["customerID"] == "C004", "Churn"].values[0] == 1
    assert cleaned.loc[cleaned["customerID"] == "C001", "Churn"].values[0] == 0


def test_clean_data_invalid_target_label():
    """Test ValueError is raised when unexpected target labels exist."""
    df_invalid = pd.DataFrame(
        {
            "TotalCharges": ["100.0"],
            "tenure": [1],
            "Churn": ["UnknownLabel"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected labels found in target column"):
        clean_data(df_invalid)


def test_separate_features_target_excludes_id_and_target(synthetic_raw_df: pd.DataFrame):
    """Test that X excludes customerID and Churn, while y is the target series."""
    cleaned = clean_data(synthetic_raw_df)
    X, y = separate_features_target(cleaned)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert "customerID" not in y.name
    assert y.name == "Churn"
    assert len(X) == len(y) == len(synthetic_raw_df)


def test_get_feature_types(synthetic_raw_df: pd.DataFrame):
    """Test automatic detection of numerical and categorical columns."""
    cleaned = clean_data(synthetic_raw_df)
    X, _ = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    # TotalCharges, MonthlyCharges, tenure, SeniorCitizen are numeric
    assert "tenure" in num_cols
    assert "MonthlyCharges" in num_cols
    assert "TotalCharges" in num_cols
    assert "Contract" in cat_cols
    assert "PaymentMethod" in cat_cols
    assert len(num_cols) + len(cat_cols) == X.shape[1]


def test_create_train_test_split_sizes_and_stratification():
    """Test 80/20 train/test split and target stratification ratio."""
    # Create balanced 100-sample dataset (30 churners, 70 non-churners)
    n_samples = 100
    y_synth = pd.Series([1] * 30 + [0] * 70, name="Churn")
    X_synth = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, size=n_samples),
            "MonthlyCharges": np.random.uniform(20, 100, size=n_samples),
        }
    )

    X_train, X_test, y_train, y_test = create_train_test_split(
        X_synth, y_synth, test_size=0.20, random_state=42, stratify=True
    )

    # Size verification
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Stratification verification (30% churn in both train and test)
    train_churn_pct = y_train.mean()
    test_churn_pct = y_test.mean()
    assert pytest.approx(train_churn_pct, abs=0.02) == 0.30
    assert pytest.approx(test_churn_pct, abs=0.02) == 0.30


def test_preprocessor_fitting_and_unseen_categories(synthetic_raw_df: pd.DataFrame):
    """Test preprocessor fits on training data and handles unseen categories during inference."""
    cleaned = clean_data(synthetic_raw_df)
    X, _ = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    assert isinstance(preprocessor, ColumnTransformer)

    # Fit on training data
    X_transformed = preprocessor.fit_transform(X)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(X)
    assert not np.isnan(X_transformed).any()

    # Create unseen inference customer with completely new categorical levels
    unseen_row = X.iloc[[0]].copy()
    unseen_row["Contract"] = "Decade-to-Decade"  # Unseen contract
    unseen_row["PaymentMethod"] = "Cryptocurrency"  # Unseen payment method

    # Transforming unseen row should not raise KeyError or ValueError
    unseen_transformed = preprocessor.transform(unseen_row)
    assert unseen_transformed.shape[0] == 1
    assert unseen_transformed.shape[1] == X_transformed.shape[1]


def test_leakage_prevention_train_only_fitting():
    """Verify preprocessor stats (median, scale) are computed solely from train split."""
    # Train set has MonthlyCharges with median = 50.0
    X_train = pd.DataFrame(
        {
            "MonthlyCharges": [40.0, 50.0, 60.0],
            "gender": ["Female", "Male", "Female"],
        }
    )
    # Test set has a missing value and very different distribution
    X_test = pd.DataFrame(
        {
            "MonthlyCharges": [np.nan, 200.0],
            "gender": ["Female", "Male"],
        }
    )

    preprocessor = build_preprocessor(["MonthlyCharges"], ["gender"])
    preprocessor.fit(X_train)

    # Imputer median on MonthlyCharges should be 50.0 (from X_train only)
    fitted_num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert fitted_num_imputer.statistics_[0] == 50.0

    # Transforming X_test must succeed using the training median (50.0) without error
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_test_trans).any()
