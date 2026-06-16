"""OR-logic aggregation and diagnostic performance evaluation."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def aggregate_or_scores(x_data: pd.DataFrame, svm_details: Dict[str, dict]) -> np.ndarray:
    """Return the maximum single-miRNA SVM decision score for each sample."""
    scores = []
    for mirna in x_data.columns:
        params = svm_details[mirna]
        z_value = (x_data[mirna].to_numpy(dtype=float) - params["mean"]) / params["std"]
        scores.append(params["w"] * z_value + params["b"])
    return np.vstack(scores).T.max(axis=1)


def validate_or_logic(
    x_data: pd.DataFrame,
    y_true: pd.Series,
    svm_details: Dict[str, dict],
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Classify a sample as diseased when any single-miRNA SVM score is non-negative."""
    y_score = aggregate_or_scores(x_data, svm_details)
    y_pred = (y_score >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    predictions = pd.DataFrame(
        {"Observed": y_true.to_numpy(), "Predicted": y_pred, "OR_score": y_score},
        index=y_true.index,
    )
    return predictions, metrics
