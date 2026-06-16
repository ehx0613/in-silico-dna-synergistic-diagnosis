"""Single-miRNA linear SVM threshold determination."""

from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_single_svm_thresholds(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Train one balanced linear SVM per miRNA and recover thresholds in log2 space."""
    thresholds: Dict[str, float] = {}
    details: Dict[str, dict] = {}

    for mirna in x_train.columns:
        values = x_train[[mirna]].values.astype(float)
        scaler = StandardScaler()
        standardised = scaler.fit_transform(values)

        svm = SVC(kernel="linear", class_weight="balanced", random_state=random_state)
        svm.fit(standardised, y_train)

        w = float(svm.coef_[0][0])
        b = float(svm.intercept_[0])
        mean = float(scaler.mean_[0])
        std = float(scaler.scale_[0])
        z_threshold = -b / w
        x_threshold = z_threshold * std + mean

        thresholds[mirna] = x_threshold
        details[mirna] = {
            "w": w,
            "b": b,
            "mean": mean,
            "std": std,
            "z_threshold": z_threshold,
            "x_threshold": x_threshold,
            "positive_direction": ">=" if w > 0 else "<=",
        }

    return thresholds, details


def threshold_table(details: Dict[str, dict]) -> pd.DataFrame:
    """Convert fitted SVM parameters to a publication-ready table."""
    rows = []
    for mirna, values in details.items():
        rows.append({"miRNA": mirna, **values})
    return pd.DataFrame(rows)
