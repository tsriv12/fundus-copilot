# Stage 2 Multimodal Fusion Summary

## Project

Self-Correcting Multi-Modal Fundus Screening Copilot Agent

## Stage 2 Goal

The goal of Stage 2 was to move beyond an image-only diabetic retinopathy baseline and fuse:

- fundus image features
- structured patient/context metadata

The model uses an EfficientNet-B0 image encoder, a small metadata MLP, feature concatenation-based fusion, and a binary prediction head.

## Dataset and Splits

The BRSET fundus dataset was used for binary diabetic retinopathy prediction.

Patient-level train/validation/test splits were used to avoid leakage.

| Split | Rows |
|---|---:|
| Train | 11,395 |
| Validation | 2,434 |
| Test | 2,437 |

The positive class is highly imbalanced, so accuracy alone is not sufficient for evaluation.

## Metadata Features Used

The first multimodal model used the following metadata inputs:

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

High-missingness fields such as insulin use, diabetes duration, and comorbidities were excluded.

## Image-Only Baseline

The image-only baseline used EfficientNet-B0 trained for 5 epochs with class imbalance handling.

Best exploratory threshold result:

| Metric | Image-Only |
|---|---:|
| Threshold | 0.70 |
| Accuracy | 0.9557 |
| Precision | 0.6256 |
| Recall | 0.7771 |
| F1 | 0.6932 |
| ROC-AUC | 0.9561 |
| PR-AUC | 0.7699 |
| False Positives | 73 |
| False Negatives | 35 |
| True Positives | 122 |
| True Negatives | 2207 |

## Multimodal Model

The multimodal model used:

- EfficientNet-B0 image encoder
- metadata MLP encoder
- concatenation-based fusion
- binary prediction head

The model checkpoint was selected using best validation loss.

## Threshold Selection

To avoid tuning directly on the test set, the operating threshold was selected on the validation set.

Selected validation threshold:

```text
0.85
```

This threshold was then applied unchanged to the held-out test set.

## Final Multimodal Test Result

| Metric | Multimodal |
|---|---:|
| Threshold | 0.85 |
| Accuracy | 0.9647 |
| Precision | 0.6983 |
| Recall | 0.7962 |
| F1 | 0.7440 |
| ROC-AUC | 0.9682 |
| PR-AUC | 0.8012 |
| False Positives | 54 |
| False Negatives | 32 |
| True Positives | 125 |
| True Negatives | 2226 |

## Model Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Image-only EfficientNet-B0 | 0.70 | 0.9557 | 0.6256 | 0.7771 | 0.6932 | 0.9561 | 0.7699 | 73 | 35 |
| Multimodal EfficientNet-B0 + Metadata | 0.85 | 0.9647 | 0.6983 | 0.7962 | 0.7440 | 0.9682 | 0.8012 | 54 | 32 |

## Key Finding

The multimodal model improved over the image-only baseline after validation-based threshold selection.

Compared with the image-only baseline, the multimodal model improved:

- Precision: 0.6256 to 0.6983
- Recall: 0.7771 to 0.7962
- F1: 0.6932 to 0.7440
- ROC-AUC: 0.9561 to 0.9682
- PR-AUC: 0.7699 to 0.8012
- False positives: 73 to 54
- False negatives: 35 to 32

This supports the project assumption that structured patient/context metadata can improve fundus screening performance when fused with image features.

## Current Stage Status

Stage 2 core modeling is complete:

- multimodal data pipeline created
- multimodal patient-level splits created
- fusion model implemented
- multimodal training completed
- validation-based threshold selection completed
- test evaluation completed
- image-only vs multimodal comparison completed

## Remaining Work

The next stage is the self-correction/auditor layer.

Planned Stage 3 work:

- flag low-confidence predictions
- flag poor-quality or artifact-heavy images
- flag cases near the decision threshold
- generate final screening output with review recommendations
- compare multimodal-only output against multimodal + auditor output
