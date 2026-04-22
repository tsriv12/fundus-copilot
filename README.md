# Self-Correcting Multi-Modal Fundus Screening Copilot Agent

This project builds a fundus screening prototype for diabetic retinopathy prediction using multimodal deep learning.

## Project Status

| Stage | Status | Summary |
|---|---|---|
| Stage 1 | Complete | Data ingestion, validation, binary DR dataset creation, patient-level splits, image-only EfficientNet-B0 baseline |
| Stage 2 | Complete | Multimodal image + metadata fusion model, validation-selected threshold, test evaluation |
| Stage 3 | Not started | Self-correction/auditor layer |
| Stage 4 | Not started | Final comparison, saved-vs-missed examples, report, and presentation |

## Dataset

The project uses the BRSET fundus dataset. The binary prediction target is:

    diabetic_retinopathy

The dataset was split at the patient level to avoid leakage.

| Split | Rows | Patients | Positive Cases |
|---|---:|---:|---:|
| Train | 11,395 | 5,966 | 754 |
| Validation | 2,434 | 1,279 | 159 |
| Test | 2,437 | 1,279 | 157 |

## Stage 1: Image-Only Baseline

The image-only baseline uses EfficientNet-B0 with class imbalance handled through positive class weighting.

Best image-only operating point from threshold sweep:

| Metric | Value |
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

## Stage 2: Multimodal Fusion Baseline

The multimodal model combines:

- EfficientNet-B0 image encoder
- metadata MLP encoder
- concatenation-based feature fusion
- binary diabetic retinopathy classification head

Metadata fields used:

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

High-missingness fields such as insulin use, diabetes duration, and comorbidities were excluded from the first fusion pass.

### Threshold Selection

The multimodal operating threshold was selected on the validation set by maximizing F1. The selected threshold was:

    0.85

This threshold was then applied unchanged to the held-out test set.

### Final Model Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Image-only EfficientNet-B0 | 0.70 | 0.9557 | 0.6256 | 0.7771 | 0.6932 | 0.9561 | 0.7699 | 73 | 35 |
| Multimodal EfficientNet-B0 + Metadata | 0.85 | 0.9647 | 0.6983 | 0.7962 | 0.7440 | 0.9682 | 0.8012 | 54 | 32 |

### Key Finding

The multimodal model improved over the image-only baseline after validation-based threshold selection.

Compared with the image-only model, the multimodal model improved:

- Precision: 0.6256 to 0.6983
- Recall: 0.7771 to 0.7962
- F1: 0.6932 to 0.7440
- ROC-AUC: 0.9561 to 0.9682
- PR-AUC: 0.7699 to 0.8012
- False positives: 73 to 54
- False negatives: 35 to 32

This supports the project assumption that structured patient/context metadata can improve fundus screening performance when fused with image features.

## Repository Structure

    configs/                   configuration files
    data/                      raw and processed data, excluded from Git
    notebooks/                 exploratory notebooks
    outputs/checkpoints/       trained model checkpoints, excluded from Git
    outputs/figures/           generated result plots
    outputs/reports/           generated reports and prediction summaries
    scripts/                   preprocessing, training, evaluation, plotting scripts
    src/data/                  dataset classes and transforms
    src/models/                model definitions
    src/training/              training utilities
    src/evaluation/            evaluation utilities
    src/inference/             inference utilities
    src/agents/                future auditor/self-correction logic
    tests/                     tests

## Important Stage 2 Artifacts

    outputs/reports/stage2_multimodal_summary.md
    outputs/reports/model_comparison_summary.csv
    outputs/reports/multimodal_test_predictions.csv
    outputs/figures/stage2_model_metric_comparison.png
    outputs/figures/stage2_error_comparison.png

## Important Checkpoints

    outputs/checkpoints/baseline_efficientnet_b0_5epochs.pt
    outputs/checkpoints/multimodal_efficientnet_b0_5epochs.pt

Model checkpoints and raw data are not intended to be committed to Git because of size and reproducibility concerns.

## Reproduce Stage 2 Results

Activate the project environment:

    cd ~/projects/radiology-copilot
    source .venv/bin/activate

Run multimodal evaluation:

    python scripts/evaluate_multimodal_baseline.py

Generate model comparison plots:

    python scripts/plot_stage2_results.py

## Next Work: Stage 3 Auditor Layer

The next stage will add a self-correction/auditor layer that flags cases for review based on:

- low confidence
- poor image quality or artifacts
- near-threshold predictions
- risky false-negative patterns
- disagreement between model confidence and metadata/context signals

The goal is to produce a final screening output with both prediction and review recommendation.
