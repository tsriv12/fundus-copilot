import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "dr_multimodal_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "dr_multimodal_val.csv")
TEST_PATH = os.path.join(DATA_DIR, "dr_multimodal_test.csv")

TARGET_COL = "diabetic_retinopathy"

METADATA_FEATURES = [
    "patient_age",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "artifacts",
]


def load_split(path):
    df = pd.read_csv(path)
    missing = [c for c in METADATA_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def evaluate(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    X_train = train_df[METADATA_FEATURES]
    y_train = train_df[TARGET_COL].astype(int).values

    X_val = val_df[METADATA_FEATURES]
    y_val = val_df[TARGET_COL].astype(int).values

    X_test = test_df[METADATA_FEATURES]
    y_test = test_df[TARGET_COL].astype(int).values

    numeric_features = METADATA_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)
    val_results = [evaluate(y_val, val_prob, t) for t in thresholds]
    val_results_df = pd.DataFrame(val_results)

    best_idx = val_results_df["f1"].idxmax()
    best_threshold = float(val_results_df.loc[best_idx, "threshold"])

    test_prob = model.predict_proba(X_test)[:, 1]
    test_result = evaluate(y_test, test_prob, best_threshold)

    val_results_df.to_csv(
        os.path.join(OUT_DIR, "metadata_logistic_regression_val_threshold_sweep.csv"),
        index=False,
    )

    test_predictions = test_df.copy()
    test_predictions["metadata_lr_probability"] = test_prob
    test_predictions["metadata_lr_prediction"] = (test_prob >= best_threshold).astype(int)
    test_predictions.to_csv(
        os.path.join(OUT_DIR, "metadata_logistic_regression_test_predictions.csv"),
        index=False,
    )

    summary = {
        "model": "Metadata-only Logistic Regression",
        "features": METADATA_FEATURES,
        "best_threshold_selected_on_validation": best_threshold,
        "test_metrics": test_result,
    }

    with open(os.path.join(OUT_DIR, "metadata_logistic_regression_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    summary_md = f"""# Metadata-only Logistic Regression Baseline

This baseline was added to satisfy the simple non-deep-learning classification baseline requirement.

## Model

- Input: 10 structured metadata features
- Target: diabetic_retinopathy
- Split: existing patient-level train/validation/test split
- Preprocessing: median imputation + standard scaling
- Class imbalance handling: class_weight="balanced"
- Threshold selection: validation F1 maximization

## Features

{chr(10).join([f"- {x}" for x in METADATA_FEATURES])}

## Validation-selected threshold

{best_threshold:.2f}

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | {test_result["accuracy"]:.4f} |
| Precision | {test_result["precision"]:.4f} |
| Recall | {test_result["recall"]:.4f} |
| F1 | {test_result["f1"]:.4f} |
| ROC-AUC | {test_result["roc_auc"]:.4f} |
| PR-AUC | {test_result["pr_auc"]:.4f} |
| True negatives | {test_result["tn"]} |
| False positives | {test_result["fp"]} |
| False negatives | {test_result["fn"]} |
| True positives | {test_result["tp"]} |

## Interpretation

The metadata-only logistic regression model is intentionally simple. It is not expected to outperform image-based deep learning models. Its purpose is to provide a transparent non-deep-learning baseline and establish how much signal exists in structured metadata alone before using fundus images or multimodal fusion.
"""

    with open(os.path.join(OUT_DIR, "metadata_logistic_regression_summary.md"), "w") as f:
        f.write(summary_md)

    print("Saved:")
    print("outputs/reports/metadata_logistic_regression_val_threshold_sweep.csv")
    print("outputs/reports/metadata_logistic_regression_test_predictions.csv")
    print("outputs/reports/metadata_logistic_regression_summary.json")
    print("outputs/reports/metadata_logistic_regression_summary.md")
    print()
    print("Best validation threshold:", best_threshold)
    print("Test metrics:")
    for k, v in test_result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
