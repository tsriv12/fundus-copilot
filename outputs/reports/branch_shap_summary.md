# Branch-Level SHAP Summary

## Purpose

This analysis calculates SHAP values on the fused latent feature space of the trained multimodal model. EfficientNet-B0 image features and metadata-MLP features are extracted from the trained model, then SHAP is applied to the final fusion classifier. Mean absolute SHAP values are averaged across the explained test subset.

## Configuration

- Background samples: 64
- Explained test samples: 128
- EfficientNet-B0 latent image features: 1280
- Metadata MLP latent features: 32

## Branch-level average SHAP result

| branch                        |   num_features |   avg_mean_abs_shap_per_feature |   total_mean_abs_shap |   max_single_feature_mean_abs_shap |   relative_total_importance |
|:------------------------------|---------------:|--------------------------------:|----------------------:|-----------------------------------:|----------------------------:|
| efficientnet_b0_image_encoder |           1280 |                      0.00323481 |             4.14055   |                          0.0174088 |                   0.978658  |
| metadata_mlp_encoder          |             32 |                      0.00282174 |             0.0902958 |                          0.0124624 |                   0.0213422 |

## Top EfficientNet-B0 latent features by mean absolute SHAP

| branch                        | feature                   |   mean_abs_shap |
|:------------------------------|:--------------------------|----------------:|
| efficientnet_b0_image_encoder | efficientnet_feature_74   |       0.0174088 |
| efficientnet_b0_image_encoder | efficientnet_feature_1117 |       0.016524  |
| efficientnet_b0_image_encoder | efficientnet_feature_1148 |       0.0140336 |
| efficientnet_b0_image_encoder | efficientnet_feature_308  |       0.0134768 |
| efficientnet_b0_image_encoder | efficientnet_feature_878  |       0.0133756 |
| efficientnet_b0_image_encoder | efficientnet_feature_558  |       0.013147  |
| efficientnet_b0_image_encoder | efficientnet_feature_389  |       0.0130431 |
| efficientnet_b0_image_encoder | efficientnet_feature_1123 |       0.0118062 |
| efficientnet_b0_image_encoder | efficientnet_feature_562  |       0.0117859 |
| efficientnet_b0_image_encoder | efficientnet_feature_682  |       0.0115869 |
| efficientnet_b0_image_encoder | efficientnet_feature_618  |       0.0115285 |
| efficientnet_b0_image_encoder | efficientnet_feature_907  |       0.0113434 |
| efficientnet_b0_image_encoder | efficientnet_feature_1065 |       0.0111576 |
| efficientnet_b0_image_encoder | efficientnet_feature_358  |       0.0111412 |
| efficientnet_b0_image_encoder | efficientnet_feature_945  |       0.0103961 |

## Top metadata MLP latent features by mean absolute SHAP

| branch               | feature                 |   mean_abs_shap |
|:---------------------|:------------------------|----------------:|
| metadata_mlp_encoder | metadata_mlp_feature_22 |      0.0124624  |
| metadata_mlp_encoder | metadata_mlp_feature_10 |      0.0124531  |
| metadata_mlp_encoder | metadata_mlp_feature_29 |      0.00964417 |
| metadata_mlp_encoder | metadata_mlp_feature_26 |      0.00789297 |
| metadata_mlp_encoder | metadata_mlp_feature_31 |      0.00663718 |
| metadata_mlp_encoder | metadata_mlp_feature_8  |      0.00627979 |
| metadata_mlp_encoder | metadata_mlp_feature_4  |      0.00516748 |
| metadata_mlp_encoder | metadata_mlp_feature_16 |      0.00451112 |
| metadata_mlp_encoder | metadata_mlp_feature_20 |      0.0042433  |
| metadata_mlp_encoder | metadata_mlp_feature_5  |      0.00396758 |
| metadata_mlp_encoder | metadata_mlp_feature_15 |      0.00372505 |
| metadata_mlp_encoder | metadata_mlp_feature_21 |      0.00362844 |
| metadata_mlp_encoder | metadata_mlp_feature_30 |      0.00357912 |
| metadata_mlp_encoder | metadata_mlp_feature_17 |      0.00257995 |
| metadata_mlp_encoder | metadata_mlp_feature_0  |      0.00166031 |

## Interpretation

The table reports average absolute SHAP values for the EfficientNet-B0 image branch and the metadata MLP branch. The total mean absolute SHAP value gives a branch-level estimate of how much each branch contributes to the final fused classifier over the explained test subset. This is a latent-feature explanation rather than a pixel-level medical heatmap.
