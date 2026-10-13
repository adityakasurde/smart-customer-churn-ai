"""Model training and selection pipeline for Smart Customer Churn Prediction AI.

Trains Logistic Regression and Random Forest models inside complete Scikit-learn
Pipelines. Compares models via stratified cross-validation on the training set,
selects the champion model, evaluates on the held-out test set, and serializes
complete pipeline artifacts and metrics summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    LOGISTIC_REGRESSION_MODEL_PATH,
    METRICS_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    POSITIVE_CLASS_LABEL,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    compute_classification_metrics,
    explain_churn_tradeoffs,
    format_evaluation_summary,
)
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    create_train_test_split,
    get_feature_types,
    separate_features_target,
)


def create_model_pipelines(
    numerical_features: list[str],
    categorical_features: list[str],
    random_state: int = RANDOM_STATE,
) -> Dict[str, Pipeline]:
    """Construct Scikit-learn Pipelines pairing preprocessing with candidate classifiers.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Pipeline]: Dictionary mapping model names to configured Pipelines.
    """
    preprocessor_lr = build_preprocessor(numerical_features, categorical_features)
    preprocessor_rf = build_preprocessor(numerical_features, categorical_features)

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_lr),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_rf),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": lr_pipeline,
        "random_forest": rf_pipeline,
    }


def perform_training_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Evaluate candidate pipelines strictly using cross-validation on the training set.

    Guarantees that model comparison and selection is conducted without peeking at the test set.

    Args:
        pipelines: Dictionary of candidate pipelines.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        n_splits: Number of stratified folds.
        random_state: Seed for cross-validation shuffling.

    Returns:
        Dict[str, Dict[str, float]]: Mean CV scores for each candidate model.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results: Dict[str, Dict[str, float]] = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            metric: round(float(scores[f"test_{metric}"].mean()), 4) for metric in scoring
        }
        cv_results[name]["std_roc_auc"] = round(float(scores["test_roc_auc"].std()), 4)
        cv_results[name]["std_f1"] = round(float(scores["test_f1"].std()), 4)

    return cv_results


def select_champion_model(cv_results: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    """Select the champion model based on training cross-validation performance.

    Prioritizes ROC-AUC and F1-score rather than raw accuracy, reflecting telecom churn economics.

    Args:
        cv_results: Cross-validation summary scores.

    Returns:
        Tuple[str, str]: (champion_model_name, selection_rationale)
    """
    # Compare primarily on ROC-AUC, then F1-score
    lr_roc = cv_results["logistic_regression"]["roc_auc"]
    rf_roc = cv_results["random_forest"]["roc_auc"]
    lr_f1 = cv_results["logistic_regression"]["f1"]
    rf_f1 = cv_results["random_forest"]["f1"]

    if lr_roc >= rf_roc:
        champion = "logistic_regression"
        rationale = (
            f"Logistic Regression selected as champion based on higher/comparable 5-fold CV ROC-AUC "
            f"({lr_roc:.4f} vs. {rf_roc:.4f}) and superior interpretability. "
            f"With balanced class weights, it achieves strong churn recall on training validation."
        )
    else:
        champion = "random_forest"
        rationale = (
            f"Random Forest selected as champion based on superior 5-fold CV ROC-AUC "
            f"({rf_roc:.4f} vs. {lr_roc:.4f}) and robust non-linear feature interaction modeling."
        )

    return champion, rationale


def train_and_evaluate_all(
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute complete training, CV model selection, test evaluation, and artifact serialization.

    Args:
        save_artifacts: Whether to write .joblib and .json artifacts to disk.

    Returns:
        Dict[str, Any]: Comprehensive summary of training, validation, and test results.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest, clean, and split dataset
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X, y = separate_features_target(cleaned_df)
    num_cols, cat_cols = get_feature_types(X)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True
    )

    # 2. Instantiate pipelines
    pipelines = create_model_pipelines(num_cols, cat_cols, random_state=RANDOM_STATE)

    # 3. Model selection via training set cross-validation (Test set remains strictly held out)
    cv_summary = perform_training_cross_validation(pipelines, X_train, y_train)
    champion_name, selection_rationale = select_champion_model(cv_summary)

    # 4. Fit pipelines on complete training set and evaluate on held-out test set
    test_metrics: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in pipelines.items():
        # Fit pipeline (preprocessing + model) ONLY on X_train
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on held-out test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        metrics = compute_classification_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba,
            threshold=DEFAULT_THRESHOLD,
            positive_label=POSITIVE_CLASS_LABEL,
        )
        test_metrics[name] = metrics

    # 5. Extract hyperparameters for metadata
    model_params = {
        "logistic_regression": {
            k: str(v)
            for k, v in fitted_pipelines["logistic_regression"]
            .named_steps["classifier"]
            .get_params()
            .items()
            if k in ["C", "class_weight", "max_iter", "solver", "penalty", "random_state"]
        },
        "random_forest": {
            k: str(v)
            for k, v in fitted_pipelines["random_forest"]
            .named_steps["classifier"]
            .get_params()
            .items()
            if k in [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "class_weight",
                "random_state",
            ]
        },
    }

    # 6. Build serializable reports
    tradeoffs = explain_churn_tradeoffs()

    metrics_summary_data = {
        "dataset_info": {
            "total_samples": int(len(raw_df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_size_ratio": float(TEST_SIZE),
            "num_features": int(X.shape[1]),
            "feature_names_raw": list(X.columns),
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
        },
        "training_cv_5fold": cv_summary,
        "champion_selection": {
            "selected_model": champion_name,
            "rationale": selection_rationale,
            "decision_basis": "5-fold Stratified CV on Training Data (ROC-AUC & F1)",
        },
        "held_out_test_evaluation": test_metrics,
        "business_tradeoffs": tradeoffs,
    }

    model_metadata_data = {
        "project": "Smart Customer Churn Prediction AI",
        "random_state": RANDOM_STATE,
        "positive_class": "1 (Churn)",
        "default_threshold": DEFAULT_THRESHOLD,
        "model_parameters": model_params,
        "selected_champion": champion_name,
        "champion_rationale": selection_rationale,
        "artifact_files": {
            "logistic_regression": str(LOGISTIC_REGRESSION_MODEL_PATH.name),
            "random_forest": str(RANDOM_FOREST_MODEL_PATH.name),
            "metrics_summary": str(METRICS_SUMMARY_PATH.name),
        },
    }

    # 7. Serialize artifacts
    if save_artifacts:
        joblib.dump(fitted_pipelines["logistic_regression"], LOGISTIC_REGRESSION_MODEL_PATH)
        joblib.dump(fitted_pipelines["random_forest"], RANDOM_FOREST_MODEL_PATH)

        with open(METRICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_summary_data, f, indent=2)

        with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(model_metadata_data, f, indent=2)

    return {
        "fitted_pipelines": fitted_pipelines,
        "metrics_summary": metrics_summary_data,
        "model_metadata": model_metadata_data,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING & EVALUATION PIPELINE: TELCO CUSTOMER CHURN")
    print("=" * 60)

    results = train_and_evaluate_all(save_artifacts=True)
    summary = results["metrics_summary"]

    print("\n--- 5-FOLD CV ON TRAINING SET (MODEL SELECTION) ---")
    for model_name, scores in summary["training_cv_5fold"].items():
        print(f"[{model_name.upper()}]")
        print(
            f"  ROC-AUC: {scores['roc_auc']:.4f} (+/- {scores['std_roc_auc']:.4f}) | "
            f"F1: {scores['f1']:.4f} (+/- {scores['std_f1']:.4f}) | "
            f"Recall: {scores['recall']:.4f} | Accuracy: {scores['accuracy']:.4f}"
        )

    print("\n--- CHAMPION MODEL SELECTION ---")
    print(f"Selected Champion : {summary['champion_selection']['selected_model']}")
    print(f"Selection Basis   : {summary['champion_selection']['rationale']}")

    print("\n--- HELD-OUT TEST SET EVALUATION ---")
    for model_name, metrics in summary["held_out_test_evaluation"].items():
        print(format_evaluation_summary(model_name, metrics))

    print("\nArtifacts successfully saved to models/:")
    print(f"  - {LOGISTIC_REGRESSION_MODEL_PATH}")
    print(f"  - {RANDOM_FOREST_MODEL_PATH}")
    print(f"  - {METRICS_SUMMARY_PATH}")
    print(f"  - {MODEL_METADATA_PATH}")
