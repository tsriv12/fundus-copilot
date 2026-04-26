# Stage 3 Auditor Layer Summary
## Purpose
Stage 3 adds a transparent rule-based auditor on top of the multimodal EfficientNet-B0 + metadata model. The auditor does not retrain the model. It reviews the model probability, selected threshold, image quality, artifact level, and borderline-risk behavior to decide whether the model output can be accepted or should be sent for manual review.
## Inputs
- Model predictions: `outputs/reports/multimodal_test_predictions.csv`
- Metadata test split: `data/processed/dr_multimodal_test.csv`
- Output file: `outputs/reports/audited_multimodal_test_predictions.csv`

## Auditor rules
- Flag predictions near the selected threshold of 0.85.
- Flag model-positive high-risk cases.
- Flag model-negative cases with elevated probability >= 0.50.
- Flag cases with inadequate image quality.
- Flag rare high artifact level cases where `artifacts >= 2`.

## Overall audit result
- Total test cases: 2437
- Accepted model outputs: 1814 (74.44%)
- Sent to manual review: 623 (25.56%)

## Error concentration by audit decision

| audit_decision      |   count |   sum |       mean |
|:--------------------|--------:|------:|-----------:|
| accept_model_output |    1814 |    12 | 0.00661521 |
| manual_review       |     623 |    74 | 0.11878    |

## Risk-band distribution

| risk_band   |   count |   percent |
|:------------|--------:|----------:|
| low         |    2082 | 0.854329  |
| high        |     179 | 0.073451  |
| medium      |     176 | 0.0722199 |

## Main audit reasons

| audit_reasons                                                                           |   count |     percent |
|:----------------------------------------------------------------------------------------|--------:|------------:|
| none                                                                                    |    1814 | 0.744358    |
| poor_image_quality                                                                      |     263 | 0.10792     |
| borderline_negative_but_elevated_probability                                            |     116 | 0.0475995   |
| model_positive_high_risk                                                                |     114 | 0.0467788   |
| near_decision_threshold;model_positive_high_risk                                        |      52 | 0.0213377   |
| near_decision_threshold;borderline_negative_but_elevated_probability                    |      42 | 0.0172343   |
| near_decision_threshold;borderline_negative_but_elevated_probability;poor_image_quality |       9 | 0.00369307  |
| borderline_negative_but_elevated_probability;poor_image_quality                         |       9 | 0.00369307  |
| near_decision_threshold;model_positive_high_risk;poor_image_quality                     |       6 | 0.00246204  |
| model_positive_high_risk;poor_image_quality                                             |       6 | 0.00246204  |
| poor_image_quality;high_artifact_level                                                  |       5 | 0.0020517   |
| model_positive_high_risk;poor_image_quality;high_artifact_level                         |       1 | 0.000410341 |

## Error capture
- False negatives in test set: 32
- False negatives sent to manual review: 20
- False positives in test set: 54
- False positives sent to manual review: 54

## Interpretation
The auditor creates a practical safety layer by separating low-risk accepted outputs from cases that deserve manual review. The accepted group has a much lower observed error rate, while the manual-review group contains a much higher share of errors and uncertainty. This supports the project goal of a self-correcting fundus screening copilot rather than a raw classifier.

## Accepted false-negative review

A follow-up check was performed on false negatives that were still accepted by the auditor. There were 12 accepted false negatives. These cases had adequate image quality, normal artifact encoding, and probabilities below the elevated-risk cutoff used by the auditor. The highest accepted false-negative probability was 0.4663.

This is an important limitation: the rule-based auditor improves safety by concentrating many errors into the manual-review bucket, but it cannot catch every missed positive without making the review rules overly broad. The current design is therefore treated as a practical first-pass auditor rather than a perfect safety filter.
