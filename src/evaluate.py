"""Model evaluation module for Smart Customer Churn Prediction AI.

Calculates classification metrics, confusion matrices, ROC-AUC,
and articulates business cost trade-offs between false positives and false negatives.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_THRESHOLD, POSITIVE_CLASS_LABEL


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    positive_label: int = POSITIVE_CLASS_LABEL,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: True binary target values (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probabilities for positive class (1D array)
                 or 2D probability matrix from predict_proba.
        threshold: Classification decision threshold used for predictions.
        positive_label: Positive class identifier (default: 1 for Churn).

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, precision, recall,
                        f1, roc_auc, confusion matrix breakdown, and report.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    # Extract 1D positive class probabilities for ROC-AUC
    proba_arr = np.asarray(y_proba)
    if proba_arr.ndim == 2:
        if proba_arr.shape[1] > positive_label:
            pos_proba = proba_arr[:, positive_label]
        else:
            pos_proba = proba_arr[:, -1]
    else:
        pos_proba = proba_arr

    # Confusion matrix elements: TN, FP, FN, TP
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, pos_proba))
    except ValueError:
        roc_auc = 0.0

    report_dict = classification_report(
        y_true_arr, y_pred_arr, labels=[0, 1], output_dict=True, zero_division=0
    )

    # Clean classification report dictionary to standard Python native types
    cleaned_report = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            cleaned_report[str(k)] = {sub_k: float(sub_v) for sub_k, sub_v in v.items()}
        else:
            cleaned_report[str(k)] = float(v)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": float(threshold),
        "positive_class": int(positive_label),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "classification_report": cleaned_report,
    }


def explain_churn_tradeoffs() -> Dict[str, str]:
    """Provide a domain-grounded explanation of the False Positive vs. False Negative trade-off.

    In churn prediction:
    - False Negatives (FN) mean missing a customer who is actually leaving.
    - False Positives (FP) mean offering retention incentives to a loyal customer.

    Returns:
        Dict[str, str]: Business rationale and strategic interpretation.
    """
    return {
        "false_negatives_risk": (
            "A False Negative occurs when an at-risk customer is classified as loyal ('No Churn'). "
            "Because no retention intervention is triggered, the customer leaves permanently, "
            "resulting in loss of customer lifetime value (LTV) and high customer acquisition cost (CAC) "
            "to replace them. False Negatives are typically the most financially damaging error in telecom."
        ),
        "false_positives_risk": (
            "A False Positive occurs when a loyal customer is classified as at-risk ('Churn'). "
            "The company might spend retention budget offering discounts, promotions, or outreach. "
            "While this incurs minor unnecessary marketing costs, the customer remains with the company."
        ),
        "strategic_recommendation": (
            "Because False Negatives carry higher financial impact than False Positives, models should "
            "prioritize high Recall and ROC-AUC rather than raw Accuracy alone. Operating at a lower "
            "decision threshold or balancing class weights improves churn recall."
        ),
    }


def format_evaluation_summary(model_name: str, metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics and confusion matrix into a clean console string.

    Args:
        model_name: Human-readable model title.
        metrics: Dictionary returned by compute_classification_metrics.

    Returns:
        str: Formatted multi-line text summary.
    """
    cm = metrics["confusion_matrix"]
    lines = [
        f"============================================================",
        f"EVALUATION RESULTS: {model_name.upper()}",
        f"============================================================",
        f"  Accuracy  : {metrics['accuracy']:.4f}",
        f"  Precision : {metrics['precision']:.4f} (Positive Class: Churn)",
        f"  Recall    : {metrics['recall']:.4f} (Positive Class: Churn)",
        f"  F1-Score  : {metrics['f1_score']:.4f}",
        f"  ROC-AUC   : {metrics['roc_auc']:.4f}",
        f"  Threshold : {metrics['threshold']}",
        f"------------------------------------------------------------",
        f"  Confusion Matrix:",
        f"    [True Negative (Retained) : {cm['true_negatives']:<4}]  [False Positive (False Alarm) : {cm['false_positives']:<4}]",
        f"    [False Negative (Missed)  : {cm['false_negatives']:<4}]  [True Positive (Caught Churn) : {cm['true_positives']:<4}]",
        f"============================================================",
    ]
    return "\n".join(lines)
