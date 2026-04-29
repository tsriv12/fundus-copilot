# Metadata-only Logistic Regression Baseline

This baseline was added to satisfy the simple non-deep-learning classification baseline requirement.

## Model

- Input: 10 structured metadata features
- Target: diabetic_retinopathy
- Split: existing patient-level train/validation/test split
- Preprocessing: median imputation + standard scaling
- Class imbalance handling: class_weight="balanced"
- Threshold selection: validation F1 maximization

## Features

- patient_age
- patient_sex
- exam_eye
- diabetes
- optic_disc
- vessels
- macula
- focus
- Illuminaton
- artifacts

## Validation-selected threshold

0.74

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8925 |
| Precision | 0.3279 |
| Recall | 0.6369 |
| F1 | 0.4329 |
| ROC-AUC | 0.8764 |
| PR-AUC | 0.3809 |
| True negatives | 2075 |
| False positives | 205 |
| False negatives | 57 |
| True positives | 100 |

## Interpretation

The metadata-only logistic regression model is intentionally simple. It is not expected to outperform image-based deep learning models. Its purpose is to provide a transparent non-deep-learning baseline and establish how much signal exists in structured metadata alone before using fundus images or multimodal fusion.
