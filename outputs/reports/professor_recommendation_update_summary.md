# Professor Recommendation Update Summary

## Summary of updates

This update addresses the professor's recommendations by adding SHAP-based explainability and empirical auditor-rule validation.

## 1. SHAP values for EfficientNet-B0 and MLP branches

SHAP values were calculated on the fused latent feature space of the trained multimodal model. The EfficientNet-B0 image encoder features and metadata MLP features were extracted, and SHAP was applied to the final fusion classifier. Mean absolute SHAP values were averaged across the explained test subset.

### Branch-level SHAP result

| branch                        |   num_features |   avg_mean_abs_shap_per_feature |   total_mean_abs_shap |   max_single_feature_mean_abs_shap |   relative_total_importance |
|:------------------------------|---------------:|--------------------------------:|----------------------:|-----------------------------------:|----------------------------:|
| efficientnet_b0_image_encoder |           1280 |                      0.00323481 |             4.14055   |                          0.0174088 |                   0.978658  |
| metadata_mlp_encoder          |             32 |                      0.00282174 |             0.0902958 |                          0.0124624 |                   0.0213422 |

The EfficientNet-B0 image branch contributes most of the total SHAP importance, while the metadata MLP branch contributes a smaller but measurable amount. This matches the project behavior: the fused model is primarily image-driven, but metadata still improves performance compared with the image-only baseline.

## 2. Raw metadata input SHAP values

A second SHAP analysis was run on the raw metadata inputs passing through the trained metadata MLP branch. Mean absolute SHAP values were averaged across the explained test subset to rank metadata inputs by contribution.

| metadata_feature   |   mean_abs_shap |   relative_importance |
|:-------------------|----------------:|----------------------:|
| patient_age        |     0.288566    |           0.954883    |
| optic_disc         |     0.00542442  |           0.0179498   |
| macula             |     0.00317503  |           0.0105064   |
| patient_sex        |     0.00259659  |           0.00859228  |
| vessels            |     0.00106533  |           0.00352526  |
| exam_eye           |     0.000705411 |           0.00233425  |
| focus              |     0.000515345 |           0.00170531  |
| Illuminaton        |     0.000151627 |           0.000501745 |
| artifacts          |     4.86538e-07 |           1.60999e-06 |
| diabetes           |     0           |           0           |

The metadata SHAP result shows patient_age as the dominant metadata input. This should be interpreted carefully because patient_age is continuous while most other metadata variables are encoded as small categorical or binary values. Future improvement should standardize metadata features before retraining to make attribution comparisons more stable.

## 3. Auditor rule validation using Option A

The auditor rule was validated by experimenting with different ranges around the multimodal model decision threshold of 0.85. For each range, the analysis measured how many cases were flagged for manual review and what percentage of those flagged cases actually had diabetic retinopathy in the dataset.

### Boundary validation table

|   threshold |   margin |   lower_bound |   upper_bound |   total_cases |   flagged_cases |   flagged_pct |   accepted_cases |   accepted_pct |   actual_dr_count_in_flagged |   actual_dr_pct_in_flagged |   overall_dr_pct |   dr_enrichment_ratio |   flagged_error_rate |   accepted_error_rate |   false_negatives_flagged |   false_negatives_total |   false_negatives_capture_pct |   false_positives_flagged |   false_positives_total |   false_positives_capture_pct |
|------------:|---------:|--------------:|--------------:|--------------:|----------------:|--------------:|-----------------:|---------------:|-----------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|----------------------:|--------------------------:|------------------------:|------------------------------:|--------------------------:|------------------------:|------------------------------:|
|        0.85 |     0.03 |          0.82 |          0.88 |          2437 |              30 |     0.0123102 |             2407 |       0.98769  |                            5 |                   0.166667 |        0.0644235 |               2.58705 |             0.433333 |            0.0303282  |                         2 |                      32 |                       0.0625  |                        11 |                      54 |                      0.203704 |
|        0.85 |     0.05 |          0.8  |          0.9  |          2437 |              51 |     0.0209274 |             2386 |       0.979073 |                           14 |                   0.27451  |        0.0644235 |               4.26102 |             0.411765 |            0.0272422  |                         7 |                      32 |                       0.21875 |                        14 |                      54 |                      0.259259 |
|        0.85 |     0.07 |          0.78 |          0.92 |          2437 |              73 |     0.0299549 |             2364 |       0.970045 |                           20 |                   0.273973 |        0.0644235 |               4.25268 |             0.424658 |            0.0232657  |                         7 |                      32 |                       0.21875 |                        24 |                      54 |                      0.444444 |
|        0.85 |     0.1  |          0.75 |          0.95 |          2437 |             109 |     0.0447271 |             2328 |       0.955273 |                           34 |                   0.311927 |        0.0644235 |               4.84182 |             0.385321 |            0.0189003  |                         9 |                      32 |                       0.28125 |                        33 |                      54 |                      0.611111 |
|        0.85 |     0.15 |          0.7  |          1    |          2437 |             253 |     0.103816  |             2184 |       0.896184 |                          134 |                   0.529644 |        0.0644235 |               8.22129 |             0.249012 |            0.0105311  |                         9 |                      32 |                       0.28125 |                        54 |                      54 |                      1        |
|        0.85 |     0.2  |          0.65 |          1.05 |          2437 |             285 |     0.116947  |             2152 |       0.883053 |                          137 |                   0.480702 |        0.0644235 |               7.46159 |             0.231579 |            0.00929368 |                        12 |                      32 |                       0.375   |                        54 |                      54 |                      1        |
|        0.85 |     0.25 |          0.6  |          1.1  |          2437 |             308 |     0.126385  |             2129 |       0.873615 |                          138 |                   0.448052 |        0.0644235 |               6.95479 |             0.217532 |            0.00892438 |                        13 |                      32 |                       0.40625 |                        54 |                      54 |                      1        |

### Selected auditor range

The selected auditor range is margin ±0.15 around threshold 0.85, corresponding to probability range [0.70, 1.00]. This flags 253 of 2437 cases (10.38%). Among the flagged cases, 52.96% actually have diabetic retinopathy, compared with the overall test-set DR prevalence of 6.44%. This gives an enrichment ratio of 8.22x.

This selection provides a practical trade-off: it produces a compact manual-review bucket, strongly enriches for true DR cases, captures all false positives, and keeps the accepted-output error rate close to 1.05%.

## 4. Why Option A was implemented first

Option A was implemented because the current repository already contains the trained image-only and fused multimodal models, plus the audited multimodal prediction outputs. Option B would require a separately trained metadata-only MLP classifier with its own predicted labels. The current MLP is a branch inside the fused model, not an independent standalone classifier. Therefore, implementing Option B without retraining a separate MLP would be methodologically weak.

## Generated artifacts

- `outputs/reports/auditor_boundary_range_validation.csv`
- `outputs/reports/auditor_boundary_range_validation.md`
- `outputs/reports/branch_shap_feature_values.csv`
- `outputs/reports/branch_shap_summary.csv`
- `outputs/reports/branch_shap_summary.md`
- `outputs/reports/metadata_input_shap_summary.csv`
- `outputs/reports/metadata_input_shap_summary.md`
- `outputs/reports/professor_recommendation_update_summary.md`
- `outputs/figures/branch_level_shap_importance.png`
- `outputs/figures/metadata_input_shap_importance.png`
