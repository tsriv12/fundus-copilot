# Final Project Summary

## Project title

Self-Correcting Multi-Modal Fundus Screening Copilot Agent

## Objective

The goal of this project was to build a fundus-image screening copilot that combines retinal image data and structured metadata to predict diabetic retinopathy risk, then adds a transparent auditor layer to flag uncertain or unsafe cases for manual review.

## Completed stages

| Stage | Description | Status |
|---|---|---|
| Stage 1 | Image-only EfficientNet-B0 baseline | Completed |
| Stage 2 | Multimodal EfficientNet-B0 + metadata model | Completed |
| Stage 3 | Rule-based auditor / self-correction layer | Completed |
| Stage 4 | Final reporting artifacts and evaluation summary | Completed |

## Model comparison

| model      | threshold_selection              |   threshold |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   false_positives |   false_negatives |   true_positives |   true_negatives |
|:-----------|:---------------------------------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|------------------:|------------------:|-----------------:|-----------------:|
| image_only | threshold_sweep_test_exploratory |        0.7  |     0.9557 |      0.6256 |   0.7771 | 0.6932 |    0.9561 |   0.7699 |                73 |                35 |              122 |             2207 |
| multimodal | validation_selected_f1           |        0.85 |     0.9647 |      0.6983 |   0.7962 | 0.744  |    0.9682 |   0.8012 |                54 |                32 |              125 |             2226 |

## Best model

The best predictive model is the multimodal EfficientNet-B0 + metadata model using a validation-selected threshold of 0.85. It improved over the image-only baseline across accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false positives, and false negatives.

## Auditor-layer result

- Total test cases: 2437
- Accepted model outputs: 1814 (74.44%)
- Sent to manual review: 623 (25.56%)
- Accepted-output error rate: 0.66%
- Manual-review error rate: 11.88%
- False negatives captured by auditor: 20 / 32
- False positives captured by auditor: 54 / 54

## Interpretation

The multimodal model is stronger than the image-only baseline, but the auditor layer is what turns the project into a copilot-style workflow. Instead of blindly returning every model prediction, the system separates lower-risk accepted outputs from cases that deserve manual review. The manual-review group has a much higher observed error rate, showing that the auditor concentrates uncertainty and model mistakes into a reviewable bucket.

## Key limitation

The auditor is rule-based and cannot catch every missed positive. In the current test set, 12 false negatives were still accepted because they had adequate image quality, normal artifact encoding, and low model probabilities. Catching those cases would require either a more sensitive model, additional clinical features, image-level explanation methods, or a more aggressive review policy that would increase manual-review volume.

## Final artifacts

- `outputs/reports/model_comparison_summary.csv`
- `outputs/reports/stage2_multimodal_summary.md`
- `outputs/reports/audited_multimodal_test_predictions.csv`
- `outputs/reports/stage3_auditor_summary.md`
- `outputs/reports/final_project_summary.md`
- `outputs/figures/stage2_model_metric_comparison.png`
- `outputs/figures/stage2_error_comparison.png`
- `outputs/figures/stage3_audit_decision_distribution.png`
- `outputs/figures/stage3_error_rate_by_audit_decision.png`
- `outputs/figures/stage3_risk_band_distribution.png`
