from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_COMPARISON = ROOT / "outputs" / "reports" / "model_comparison_summary.csv"
AUDITED = ROOT / "outputs" / "reports" / "audited_multimodal_test_predictions.csv"
OUT = ROOT / "outputs" / "reports" / "final_project_summary.md"


def main() -> None:
    model_df = pd.read_csv(MODEL_COMPARISON)
    audit_df = pd.read_csv(AUDITED)

    total = len(audit_df)
    accepted = int((audit_df["audit_decision"] == "accept_model_output").sum())
    reviewed = int((audit_df["audit_decision"] == "manual_review").sum())

    accepted_error_rate = audit_df.loc[
        audit_df["audit_decision"] == "accept_model_output", "is_error"
    ].mean()

    review_error_rate = audit_df.loc[
        audit_df["audit_decision"] == "manual_review", "is_error"
    ].mean()

    false_negatives = audit_df[
        (audit_df["label"].astype(int) == 1)
        & (audit_df["prediction_threshold_0_85"].astype(int) == 0)
    ]

    false_positives = audit_df[
        (audit_df["label"].astype(int) == 0)
        & (audit_df["prediction_threshold_0_85"].astype(int) == 1)
    ]

    flagged_fn = int((false_negatives["audit_decision"] == "manual_review").sum())
    flagged_fp = int((false_positives["audit_decision"] == "manual_review").sum())

    lines = []

    lines.append("# Final Project Summary\n\n")
    lines.append("## Project title\n\n")
    lines.append("Self-Correcting Multi-Modal Fundus Screening Copilot Agent\n\n")

    lines.append("## Objective\n\n")
    lines.append(
        "The goal of this project was to build a fundus-image screening copilot "
        "that combines retinal image data and structured metadata to predict "
        "diabetic retinopathy risk, then adds a transparent auditor layer to flag "
        "uncertain or unsafe cases for manual review.\n\n"
    )

    lines.append("## Completed stages\n\n")
    lines.append("| Stage | Description | Status |\n")
    lines.append("|---|---|---|\n")
    lines.append("| Stage 1 | Image-only EfficientNet-B0 baseline | Completed |\n")
    lines.append("| Stage 2 | Multimodal EfficientNet-B0 + metadata model | Completed |\n")
    lines.append("| Stage 3 | Rule-based auditor / self-correction layer | Completed |\n")
    lines.append("| Stage 4 | Final reporting artifacts and evaluation summary | In progress |\n\n")

    lines.append("## Model comparison\n\n")
    lines.append(model_df.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Best model\n\n")
    lines.append(
        "The best predictive model is the multimodal EfficientNet-B0 + metadata "
        "model using a validation-selected threshold of 0.85. It improved over "
        "the image-only baseline across accuracy, precision, recall, F1, ROC-AUC, "
        "PR-AUC, false positives, and false negatives.\n\n"
    )

    lines.append("## Auditor-layer result\n\n")
    lines.append(f"- Total test cases: {total}\n")
    lines.append(f"- Accepted model outputs: {accepted} ({accepted / total:.2%})\n")
    lines.append(f"- Sent to manual review: {reviewed} ({reviewed / total:.2%})\n")
    lines.append(f"- Accepted-output error rate: {accepted_error_rate:.2%}\n")
    lines.append(f"- Manual-review error rate: {review_error_rate:.2%}\n")
    lines.append(f"- False negatives captured by auditor: {flagged_fn} / {len(false_negatives)}\n")
    lines.append(f"- False positives captured by auditor: {flagged_fp} / {len(false_positives)}\n\n")

    lines.append("## Interpretation\n\n")
    lines.append(
        "The multimodal model is stronger than the image-only baseline, but the "
        "auditor layer is what turns the project into a copilot-style workflow. "
        "Instead of blindly returning every model prediction, the system separates "
        "lower-risk accepted outputs from cases that deserve manual review. The "
        "manual-review group has a much higher observed error rate, showing that "
        "the auditor concentrates uncertainty and model mistakes into a reviewable bucket.\n\n"
    )

    lines.append("## Key limitation\n\n")
    lines.append(
        "The auditor is rule-based and cannot catch every missed positive. In the "
        "current test set, 12 false negatives were still accepted because they had "
        "adequate image quality, normal artifact encoding, and low model probabilities. "
        "Catching those cases would require either a more sensitive model, additional "
        "clinical features, image-level explanation methods, or a more aggressive review "
        "policy that would increase manual-review volume.\n\n"
    )

    lines.append("## Final artifacts\n\n")
    lines.append("- `outputs/reports/model_comparison_summary.csv`\n")
    lines.append("- `outputs/reports/stage2_multimodal_summary.md`\n")
    lines.append("- `outputs/reports/audited_multimodal_test_predictions.csv`\n")
    lines.append("- `outputs/reports/stage3_auditor_summary.md`\n")
    lines.append("- `outputs/reports/final_project_summary.md`\n")
    lines.append("- `outputs/figures/stage2_model_metric_comparison.png`\n")
    lines.append("- `outputs/figures/stage2_error_comparison.png`\n")
    lines.append("- `outputs/figures/stage3_audit_decision_distribution.png`\n")
    lines.append("- `outputs/figures/stage3_error_rate_by_audit_decision.png`\n")
    lines.append("- `outputs/figures/stage3_risk_band_distribution.png`\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

