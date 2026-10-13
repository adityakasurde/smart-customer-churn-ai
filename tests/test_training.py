"""Unit tests for the model training and pipeline serialization workflow."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.evaluate import compute_classification_metrics
from src.preprocessing import clean_data, get_feature_types, separate_features_target
from src.train import create_model_pipelines, perform_training_cross_validation, select_champion_model


@pytest.fixture
def synthetic_training_df() -> pd.DataFrame:
    """Create a self-contained synthetic dataset for training unit tests without network access."""
    np.random.seed(42)
    n = 60
    tenure_vals = np.random.randint(0, 72, size=n)
    monthly_vals = np.random.uniform(20.0, 100.0, size=n)
    total_vals = [
        " " if t == 0 else str(round(t * m, 2))
        for t, m in zip(tenure_vals, monthly_vals)
    ]
    churn_vals = np.random.choice(["No", "Yes"], size=n, p=[0.7, 0.3])

    return pd.DataFrame(
        {
            "customerID": [f"ID_{i:03d}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "Partner": np.random.choice(["Yes", "No"], size=n),
            "Dependents": np.random.choice(["Yes", "No"], size=n),
            "tenure": tenure_vals,
            "PhoneService": np.random.choice(["Yes", "No"], size=n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], size=n
            ),
            "MonthlyCharges": monthly_vals,
            "TotalCharges": total_vals,
            "Churn": churn_vals,
        }
    )


def test_pipelines_instantiation_and_fitting(synthetic_training_df: pd.DataFrame):
    """Verify both Logistic Regression and Random Forest pipelines fit successfully on raw features."""
    cleaned = clean_data(synthetic_training_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    pipelines = create_model_pipelines(num_cols, cat_cols, random_state=42)

    assert "logistic_regression" in pipelines
    assert "random_forest" in pipelines

    for name, pipe in pipelines.items():
        assert isinstance(pipe, Pipeline)
        pipe.fit(X, y)

        preds = pipe.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert len(preds) == len(X)

        probas = pipe.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)


def test_cross_validation_training_only(synthetic_training_df: pd.DataFrame):
    """Verify that training cross-validation computes valid metrics across folds."""
    cleaned = clean_data(synthetic_training_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    pipelines = create_model_pipelines(num_cols, cat_cols, random_state=42)
    cv_scores = perform_training_cross_validation(pipelines, X, y, n_splits=3, random_state=42)

    for name in ["logistic_regression", "random_forest"]:
        assert name in cv_scores
        assert 0.0 <= cv_scores[name]["accuracy"] <= 1.0
        assert 0.0 <= cv_scores[name]["roc_auc"] <= 1.0
        assert 0.0 <= cv_scores[name]["f1"] <= 1.0

    champion, rationale = select_champion_model(cv_scores)
    assert champion in ["logistic_regression", "random_forest"]
    assert len(rationale) > 20


def test_pipeline_serialization_and_inference(
    synthetic_training_df: pd.DataFrame, tmp_path: Path
):
    """Verify complete Pipeline serialization with joblib and prediction on raw unencoded test rows."""
    cleaned = clean_data(synthetic_training_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    pipelines = create_model_pipelines(num_cols, cat_cols, random_state=42)
    lr_pipe = pipelines["logistic_regression"]
    lr_pipe.fit(X, y)

    # Save artifact
    artifact_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(lr_pipe, artifact_path)
    assert artifact_path.is_file()

    # Load artifact
    loaded_pipe = joblib.load(artifact_path)
    assert isinstance(loaded_pipe, Pipeline)

    # Inference on a single unencoded raw DataFrame row
    sample_row = X.iloc[[0]].copy()
    pred = loaded_pipe.predict(sample_row)
    proba = loaded_pipe.predict_proba(sample_row)

    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_no_leakage_in_training(synthetic_training_df: pd.DataFrame):
    """Verify that preprocessor scalers and imputers are strictly learned from X_train."""
    cleaned = clean_data(synthetic_training_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = get_feature_types(X)

    # Artificial train split with specific MonthlyCharges
    X_train = X.iloc[:40].copy()
    y_train = y.iloc[:40].copy()
    X_test = X.iloc[40:].copy()

    # Calculate expected median on X_train only
    expected_median = float(X_train["MonthlyCharges"].median())

    pipelines = create_model_pipelines(num_cols, cat_cols, random_state=42)
    pipe = pipelines["logistic_regression"]
    pipe.fit(X_train, y_train)

    # Extract fitted imputer from inside the pipeline's ColumnTransformer
    preprocessor = pipe.named_steps["preprocessor"]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    mc_idx = num_cols.index("MonthlyCharges")
    fitted_median = imputer.statistics_[mc_idx]

    assert pytest.approx(fitted_median, abs=1e-4) == expected_median
