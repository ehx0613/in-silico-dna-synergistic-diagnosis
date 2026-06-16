"""Statistical screening of TCGA miRNA expression data."""

from typing import List, Tuple

import pandas as pd
from scipy.stats import levene, mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests


def adaptive_test(diseased: pd.Series, healthy: pd.Series) -> Tuple[float, str]:
    """Select Student's t-test, Welch's t-test or Mann-Whitney U test."""
    try:
        normal = (
            len(diseased) >= 3
            and len(healthy) >= 3
            and shapiro(diseased).pvalue > 0.05
            and shapiro(healthy).pvalue > 0.05
        )
    except Exception:
        normal = False

    if normal:
        equal_var = levene(diseased, healthy).pvalue > 0.05
        result = ttest_ind(diseased, healthy, equal_var=equal_var)
        return float(result.pvalue), "Student t-test" if equal_var else "Welch t-test"

    result = mannwhitneyu(diseased, healthy, alternative="two-sided")
    return float(result.pvalue), "Mann-Whitney U test"


def screen_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fdr_threshold: float = 0.05,
    delta_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply adaptive tests and Benjamini-Hochberg correction on training data."""
    diseased = x_train.loc[y_train == 1]
    healthy = x_train.loc[y_train == 0]
    rows = []

    for mirna in x_train.columns:
        p_raw, test_method = adaptive_test(diseased[mirna], healthy[mirna])
        mean_diseased = float(diseased[mirna].mean())
        mean_healthy = float(healthy[mirna].mean())
        rows.append(
            {
                "miRNA": mirna,
                "p_raw": p_raw,
                "FDR": 0.0,
                "delta_log2": mean_diseased - mean_healthy,
                "Test_Method": test_method,
                "Mean_Diseased_log2": mean_diseased,
                "Mean_Healthy_log2": mean_healthy,
            }
        )

    all_results = pd.DataFrame(rows)
    all_results["FDR"] = multipletests(
        all_results["p_raw"], alpha=fdr_threshold, method="fdr_bh"
    )[1]
    candidates = all_results[
        (all_results["FDR"] < fdr_threshold)
        & (all_results["delta_log2"].abs() > delta_threshold)
    ].sort_values("FDR").reset_index(drop=True)
    return all_results, candidates


def remove_correlated_features(
    x_train: pd.DataFrame,
    candidates: pd.DataFrame,
    corr_threshold: float = 0.8,
) -> pd.DataFrame:
    """Greedily retain the lowest-FDR feature from highly correlated pairs."""
    remaining: List[str] = candidates["miRNA"].tolist()
    selected: List[str] = []

    while remaining:
        current = remaining[0]
        selected.append(current)
        correlations = x_train[remaining[1:]].corrwith(x_train[current]).abs()
        excluded = set(correlations[correlations > corr_threshold].index)
        remaining = [name for name in remaining[1:] if name not in excluded]

    return (
        candidates[candidates["miRNA"].isin(selected)]
        .sort_values("FDR")
        .reset_index(drop=True)
    )
