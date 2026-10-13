"""Unit tests for the model evaluation module."""

import json
import numpy as np
import pytest

from src.evaluate import (
    compute_classification_metrics,
    explain_churn_tradeoffs,
    format_evaluation_summary,
)


def test_compute_classification_metrics_standard():
    """Test metric computation on known predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])  # 3 TP, 1 FN, 1 FP, 3 TN
    y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.85, 0.65, 0.75])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba, threshold=0.5)

    assert metrics["accuracy"] == 0.75  # 6/8
    assert metrics["precision"] == 0.75  # 3/4
    assert metrics["recall"] == 0.75  # 3/4
    assert metrics["f1_score"] == 0.75
    assert metrics["roc_auc"] > 0.80

    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 3
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 3
    assert cm["matrix"] == [[3, 1], [1, 3]]


def test_compute_classification_metrics_2d_proba():
    """Test metric calculation when 2D probability matrix from predict_proba is passed."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    # 2D array [prob_0, prob_1]
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_metrics_json_serializability():
    """Ensure all dictionary values are pure Python native types serializable to JSON."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7]])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    json_str = json.dumps(metrics)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["accuracy"] == 1.0


def test_explain_churn_tradeoffs():
    """Verify that domain trade-offs explain false positives and false negatives clearly."""
    tradeoffs = explain_churn_tradeoffs()
    assert "false_negatives_risk" in tradeoffs
    assert "false_positives_risk" in tradeoffs
    assert "strategic_recommendation" in tradeoffs
    assert "False Negative" in tradeoffs["false_negatives_risk"]
    assert "False Positive" in tradeoffs["false_positives_risk"]


def test_format_evaluation_summary():
    """Verify console formatting of evaluation metrics."""
    metrics = {
        "accuracy": 0.80,
        "precision": 0.65,
        "recall": 0.55,
        "f1_score": 0.59,
        "roc_auc": 0.84,
        "threshold": 0.5,
        "confusion_matrix": {
            "true_negatives": 800,
            "false_positives": 200,
            "false_negatives": 150,
            "true_positives": 250,
        },
    }
    summary = format_evaluation_summary("logistic_regression", metrics)
    assert "LOGISTIC_REGRESSION" in summary
    assert "0.8000" in summary
    assert "True Negative" in summary
