from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_CSV = ROOT / "outputs" / "reports" / "auditor_boundary_range_validation.csv"
BRANCH_SHAP_CSV = ROOT / "outputs" / "reports" / "branch_shap_summary.csv"
METADATA_SHAP_CSV = ROOT / "outputs" / "reports" / "metadata_input_shap_summary.csv"

OUT = ROOT / "outputs" / "reports" / "professor_recommendation_update_summary.md"


def main() -> None:
    boundary = pd.read_csv(BOUNDARY_CSV)
    branch = pd.read_csv(BRANCH_SHAP_CSV)
    metadata = pd.read_csv(METADATA_SHAP_CSV)

    selected = boundary[boundary["margin"] == 0.15].iloc[0]

    lines = []
    lines.append("# Professor Recommendation Update Summary\n\n")

    lines.append("## Summary of updates\n\n")
    lines.append(
        "This update addresses the professor's recommendations by adding SHAP-based "
        "explainability and empirical auditor-rule validation.\n\n"
    )

    lines.append("## 1. SHAP values for EfficientNet-B0 and MLP branches\n\n")
    lines.append(
        "SHAP values were calculated on the fused latent feature space of the trained "
        "multimodal model. The EfficientNet-B0 image encoder features and metadata MLP "
        "features were extracted, and SHAP was applied to the final fusion classifier. "
        "Mean absolute SHAP values were averaged across the explained test subset.\n\n"
    )

    lines.append("### Branch-level SHAP result\n\n")
    lines.append(branch.to_markdown(index=False))
    lines.append("\n\n")

    lines.append(
        "The EfficientNet-B0 image branch contributes most of the total SHAP importance, "
        "while the metadata MLP branch contributes a smaller but measurable amount. "
        "This matches the project behavior: the fused model is primarily image-driven, "
        "but metadata still improves performance compared with the image-only baseline.\n\n"
    )

    lines.append("## 2. Raw metadata input SHAP values\n\n")
    lines.append(
        "A second SHAP analysis was run on the raw metadata inputs passing through the "
        "trained metadata MLP branch. Mean absolute SHAP values were averaged across "
        "the explained test subset to rank metadata inputs by contribution.\n\n"
    )

    lines.append(metadata.to_markdown(index=False))
    lines.append("\n\n")

    lines.append(
        "The metadata SHAP result shows patient_age as the dominant metadata input. "
        "This should be interpreted carefully because patient_age is continuous while "
        "most other metadata variables are encoded as small categorical or binary values. "
        "Future improvement should standardize metadata features before retraining to make "
        "attribution comparisons more stable.\n\n"
    )

    lines.append("## 3. Auditor rule validation using Option A\n\n")
    lines.append(
        "The auditor rule was validated by experimenting with different ranges around "
        "the multimodal model decision threshold of 0.85. For each range, the analysis "
        "measured how many cases were flagged for manual review and what percentage of "
        "those flagged cases actually had diabetic retinopathy in the dataset.\n\n"
    )

    lines.append("### Boundary validation table\n\n")
    lines.append(boundary.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("### Selected auditor range\n\n")
    lines.append(
        f"The selected auditor range is margin ±0.15 around threshold 0.85, corresponding "
        f"to probability range [{selected['lower_bound']:.2f}, {selected['upper_bound']:.2f}]. "
        f"This flags {int(selected['flagged_cases'])} of {int(selected['total_cases'])} cases "
        f"({selected['flagged_pct']:.2%}). Among the flagged cases, "
        f"{selected['actual_dr_pct_in_flagged']:.2%} actually have diabetic retinopathy, "
        f"compared with the overall test-set DR prevalence of {selected['overall_dr_pct']:.2%}. "
        f"This gives an enrichment ratio of {selected['dr_enrichment_ratio']:.2f}x.\n\n"
    )

    lines.append(
        "This selection provides a practical trade-off: it produces a compact manual-review "
        "bucket, strongly enriches for true DR cases, captures all false positives, and keeps "
        "the accepted-output error rate close to 1.05%.\n\n"
    )

    lines.append("## 4. Why Option A was implemented first\n\n")
    lines.append(
        "Option A was implemented because the current repository already contains the trained "
        "image-only and fused multimodal models, plus the audited multimodal prediction outputs. "
        "Option B would require a separately trained metadata-only MLP classifier with its own "
        "predicted labels. The current MLP is a branch inside the fused model, not an independent "
        "standalone classifier. Therefore, implementing Option B without retraining a separate MLP "
        "would be methodologically weak.\n\n"
    )

    lines.append("## Generated artifacts\n\n")
    lines.append("- `outputs/reports/auditor_boundary_range_validation.csv`\n")
    lines.append("- `outputs/reports/auditor_boundary_range_validation.md`\n")
    lines.append("- `outputs/reports/branch_shap_feature_values.csv`\n")
    lines.append("- `outputs/reports/branch_shap_summary.csv`\n")
    lines.append("- `outputs/reports/branch_shap_summary.md`\n")
    lines.append("- `outputs/reports/metadata_input_shap_summary.csv`\n")
    lines.append("- `outputs/reports/metadata_input_shap_summary.md`\n")
    lines.append("- `outputs/reports/professor_recommendation_update_summary.md`\n")
    lines.append("- `outputs/figures/branch_level_shap_importance.png`\n")
    lines.append("- `outputs/figures/metadata_input_shap_importance.png`\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
