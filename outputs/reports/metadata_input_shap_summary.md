# Metadata Input SHAP Summary

## Purpose

This analysis calculates SHAP values for the raw metadata inputs passing through the trained metadata MLP encoder. Mean absolute SHAP values are averaged across the explained test subset to rank metadata inputs by contribution to the metadata representation.

## Configuration

- Background samples: 100
- Explained test samples: 300
- Metadata features: 10

## Average absolute SHAP by metadata feature

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

## Interpretation

Higher mean absolute SHAP values indicate metadata inputs that more strongly affect the trained metadata MLP branch. This complements the branch-level SHAP analysis, which showed the relative contribution of EfficientNet-B0 latent image features and metadata MLP latent features to the final fused classifier.
