"""Run statistical screening, SVM threshold determination and OR-logic validation."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screening = load_module("statistical_screening", "01_statistical_screening.py")
thresholds = load_module("svm_thresholds", "02_svm_thresholds.py")
validation = load_module("or_logic_validation", "03_or_logic_validation.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True, help="miRNA-by-sample RPM matrix CSV")
    parser.add_argument("--metadata", required=True, help="Metadata CSV containing SampleID and Condition")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(args.expression, index_col=0)
    metadata = pd.read_csv(args.metadata)
    metadata = metadata.set_index("SampleID" if "SampleID" in metadata.columns else metadata.columns[0])

    expression = expression.loc[(expression == 0).mean(axis=1) <= 0.10]
    expression = np.log2(expression + 1)
    common = expression.columns.intersection(metadata.index)
    x_all = expression[common].T
    y_all = metadata.loc[common, "Condition"].map({"Diseased": 1, "Healthy": 0})
    valid = y_all.notna()
    x_all, y_all = x_all.loc[valid], y_all.loc[valid]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y_all,
    )

    all_stats, candidates = screening.screen_features(x_train, y_train)
    decorrelated = screening.remove_correlated_features(x_train, candidates)
    top_features = decorrelated.head(args.top_k)
    if len(top_features) < args.top_k:
        raise RuntimeError(f"Only {len(top_features)} features remained; {args.top_k} required.")

    mirnas = top_features["miRNA"].tolist()
    _, svm_details = thresholds.train_single_svm_thresholds(
        x_train[mirnas], y_train, args.random_state
    )
    train_predictions, train_metrics = validation.validate_or_logic(
        x_train[mirnas], y_train, svm_details
    )
    test_predictions, test_metrics = validation.validate_or_logic(
        x_test[mirnas], y_test, svm_details
    )

    all_stats.to_csv(output / "01_all_statistical_tests.csv", index=False)
    candidates.to_csv(output / "02_filtered_candidates.csv", index=False)
    top_features.to_csv(output / "03_selected_top_miRNAs.csv", index=False)
    thresholds.threshold_table(svm_details).to_csv(output / "04_svm_thresholds.csv", index=False)
    train_predictions.to_csv(output / "05_training_predictions.csv")
    test_predictions.to_csv(output / "06_test_predictions.csv")
    pd.DataFrame([train_metrics, test_metrics], index=["Training", "Test"]).to_csv(
        output / "07_or_logic_metrics.csv"
    )


if __name__ == "__main__":
    main()
