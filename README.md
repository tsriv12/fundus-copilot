# Self-Correcting Multi-Modal Fundus Screening Copilot Agent

This project builds a fundus screening prototype for diabetic retinopathy prediction using multimodal deep learning.

## Project Status

| Stage | Status | Summary |
|---|---|---|
| Stage 1 | Complete | Data ingestion, validation, binary DR dataset creation, patient-level splits, image-only EfficientNet-B0 baseline |
| Stage 2 | Complete | Multimodal image + metadata fusion model, validation-selected threshold, test evaluation |
| Stage 3 | Complete | Rule-based auditor/self-correction layer with manual-review routing |
| Stage 4 | Complete | Final comparison, auditor evaluation, plots, and consolidated project summary |

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
    src/agents/                auditor/self-correction logic
    tests/                     tests

## Important Artifacts

    outputs/reports/stage2_multimodal_summary.md
    outputs/reports/model_comparison_summary.csv
    outputs/reports/multimodal_test_predictions.csv
    outputs/reports/audited_multimodal_test_predictions.csv
    outputs/reports/stage3_auditor_summary.md
    outputs/reports/final_project_summary.md
    outputs/figures/stage2_model_metric_comparison.png
    outputs/figures/stage2_error_comparison.png
    outputs/figures/stage3_audit_decision_distribution.png
    outputs/figures/stage3_error_rate_by_audit_decision.png
    outputs/figures/stage3_risk_band_distribution.png

## Important Checkpoints

    outputs/checkpoints/baseline_efficientnet_b0_5epochs.pt
    outputs/checkpoints/multimodal_efficientnet_b0_5epochs.pt

Model checkpoints and raw data are not intended to be committed to Git because of size and reproducibility concerns.

## Reproduce Results

Activate the project environment:

    cd ~/projects/radiology-copilot
    source .venv/bin/activate

Run multimodal evaluation:

    python scripts/evaluate_multimodal_baseline.py

Generate model comparison plots:

    python scripts/plot_stage2_results.py

Run the Stage 3 auditor:

    python scripts/run_auditor.py

Generate Stage 3 auditor summary:

    python scripts/summarize_auditor_results.py

Generate Stage 3 plots:

    python scripts/plot_stage3_auditor_results.py

Generate final project summary:

    python scripts/create_final_project_summary.py

## Stage 3: Auditor / Self-Correction Layer

Stage 3 adds a transparent rule-based auditor on top of the multimodal model.

The auditor reviews model probability, the selected threshold, near-threshold uncertainty, elevated probability despite negative prediction, model-positive high-risk cases, image quality, and artifact severity.

The auditor produces:

- `audit_decision`
- `audit_reasons`
- `risk_band`

### Auditor Result

| Metric | Value |
|---|---:|
| Total test cases | 2,437 |
| Accepted model outputs | 1,814 |
| Sent to manual review | 623 |
| Accepted-output share | 74.44% |
| Manual-review share | 25.56% |
| Accepted-output error rate | 0.66% |
| Manual-review error rate | 11.88% |
| False negatives captured by auditor | 20 / 32 |
| False positives captured by auditor | 54 / 54 |

The auditor concentrates risk into the manual-review group. The accepted-output group has a much lower observed error rate, while the manual-review group contains a much higher share of errors and uncertainty.

### Limitation

The rule-based auditor does not catch every missed positive. In the current test set, 12 false negatives were still accepted because they had adequate image quality, normal artifact encoding, and low model probabilities.

## Professor Recommendation Update

Additional analysis was added after professor feedback to strengthen explainability and auditor-rule design.

### SHAP Explainability

SHAP analysis was added at two levels:

1. **Branch-level SHAP on fused latent features**
   - EfficientNet-B0 image encoder latent features
   - metadata MLP encoder latent features
   - SHAP values averaged across the explained test subset

2. **Raw metadata-input SHAP**
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

Branch-level SHAP showed that the EfficientNet-B0 image branch contributed most of the total fused-model importance, while the metadata MLP branch contributed a smaller but measurable amount.

Raw metadata SHAP showed patient_age as the dominant metadata input. This result should be interpreted carefully because patient_age is continuous while most other metadata fields are categorical or binary.

### Auditor Rule Validation: Option A

The auditor rule was validated by testing different probability ranges around the multimodal model threshold of 0.85.

Tested margins:
- ±0.03
- ±0.05
- ±0.07
- ±0.10
- ±0.15
- ±0.20
- ±0.25

The selected range was:

```text
0.70 to 1.00
```

This corresponds to margin ±0.15 around the selected threshold of 0.85.

At this range:

| Metric | Value |
|---|---|
| Flagged cases | 253 / 2,437 |
| Flagged percentage | 10.38% |
| Actual DR percentage in flagged cases | 52.96% |
| Overall DR percentage in test set | 6.44% |
| DR enrichment ratio | 8.22x |
| Accepted-output error rate | 1.05% |
| False positives captured | 54 / 54 |
| False negatives captured | 9 / 32 |

This supports the auditor design because the manual-review bucket is not arbitrary. It is empirically enriched for actual diabetic retinopathy cases compared with the overall held-out test distribution.

### Why Option B Was Not Implemented

Option B would require comparing predicted labels from three independent models:

- image-only EfficientNet-B0
- metadata-only MLP
- fused multimodal model

The current project has an image-only model and a fused multimodal model. The MLP exists as a metadata branch inside the fused model, not as an independently trained metadata-only classifier. Therefore, implementing Option B without training a separate metadata-only MLP would be methodologically weak.

### Additional Artifacts

- outputs/reports/auditor_boundary_range_validation.csv
- outputs/reports/auditor_boundary_range_validation.md
- outputs/reports/branch_shap_feature_values.csv
- outputs/reports/branch_shap_summary.csv
- outputs/reports/branch_shap_summary.md
- outputs/reports/metadata_input_shap_summary.csv
- outputs/reports/metadata_input_shap_summary.md
- outputs/reports/professor_recommendation_update_summary.md
- outputs/figures/branch_level_shap_importance.png
- outputs/figures/metadata_input_shap_importance.png
