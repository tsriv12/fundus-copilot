# Auditor Boundary Range Validation

## Purpose

This analysis validates different near-threshold auditor ranges around the multimodal model decision threshold of 0.85. For each range, the script calculates what percentage of cases would be flagged for manual review and checks how many of those flagged cases actually have diabetic retinopathy in the test dataset.

## Boundary range results

|   threshold |   margin |   lower_bound |   upper_bound |   total_cases |   flagged_cases |   flagged_pct |   accepted_cases |   accepted_pct |   actual_dr_count_in_flagged |   actual_dr_pct_in_flagged |   overall_dr_pct |   dr_enrichment_ratio |   flagged_error_rate |   accepted_error_rate |   false_negatives_flagged |   false_negatives_total |   false_negatives_capture_pct |   false_positives_flagged |   false_positives_total |   false_positives_capture_pct |
|------------:|---------:|--------------:|--------------:|--------------:|----------------:|--------------:|-----------------:|---------------:|-----------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|----------------------:|--------------------------:|------------------------:|------------------------------:|--------------------------:|------------------------:|------------------------------:|
|        0.85 |     0.03 |          0.82 |          0.88 |          2437 |              30 |     0.0123102 |             2407 |       0.98769  |                            5 |                   0.166667 |        0.0644235 |               2.58705 |             0.433333 |            0.0303282  |                         2 |                      32 |                       0.0625  |                        11 |                      54 |                      0.203704 |
|        0.85 |     0.05 |          0.8  |          0.9  |          2437 |              51 |     0.0209274 |             2386 |       0.979073 |                           14 |                   0.27451  |        0.0644235 |               4.26102 |             0.411765 |            0.0272422  |                         7 |                      32 |                       0.21875 |                        14 |                      54 |                      0.259259 |
|        0.85 |     0.07 |          0.78 |          0.92 |          2437 |              73 |     0.0299549 |             2364 |       0.970045 |                           20 |                   0.273973 |        0.0644235 |               4.25268 |             0.424658 |            0.0232657  |                         7 |                      32 |                       0.21875 |                        24 |                      54 |                      0.444444 |
|        0.85 |     0.1  |          0.75 |          0.95 |          2437 |             109 |     0.0447271 |             2328 |       0.955273 |                           34 |                   0.311927 |        0.0644235 |               4.84182 |             0.385321 |            0.0189003  |                         9 |                      32 |                       0.28125 |                        33 |                      54 |                      0.611111 |
|        0.85 |     0.15 |          0.7  |          1    |          2437 |             253 |     0.103816  |             2184 |       0.896184 |                          134 |                   0.529644 |        0.0644235 |               8.22129 |             0.249012 |            0.0105311  |                         9 |                      32 |                       0.28125 |                        54 |                      54 |                      1        |
|        0.85 |     0.2  |          0.65 |          1.05 |          2437 |             285 |     0.116947  |             2152 |       0.883053 |                          137 |                   0.480702 |        0.0644235 |               7.46159 |             0.231579 |            0.00929368 |                        12 |                      32 |                       0.375   |                        54 |                      54 |                      1        |
|        0.85 |     0.25 |          0.6  |          1.1  |          2437 |             308 |     0.126385  |             2129 |       0.873615 |                          138 |                   0.448052 |        0.0644235 |               6.95479 |             0.217532 |            0.00892438 |                        13 |                      32 |                       0.40625 |                        54 |                      54 |                      1        |

## Selected range

Based on the boundary-range validation, margin ±0.15 around the selected threshold of 0.85 is the most practical auditor range. This corresponds to probability range [0.70, 1.00].

At this range, the auditor flags 253 of 2,437 test cases, or 10.38% of the dataset. Among the flagged cases, 52.96% actually have diabetic retinopathy, compared with the overall test-set DR prevalence of 6.44%. This gives an 8.22x DR enrichment in the manual-review group.

This range also captures all false positives and keeps the accepted-output error rate close to 1.05%. Wider ranges catch a few more false negatives, but they increase review volume and reduce DR enrichment. Therefore, ±0.15 is a stronger trade-off between safety, review burden, and clinical usefulness.

## Interpretation

A useful auditor range should not simply flag many cases. It should flag a reviewable subset that is enriched for actual DR cases or model errors. The boundary validation shows that the auditor range can be selected empirically rather than arbitrarily. The selected ±0.15 range provides a compact review bucket with strong enrichment for actual DR-positive cases.
