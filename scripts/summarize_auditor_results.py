from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs" / "reports" / "audited_multimodal_test_predictions.csv"
OUT_PATH = ROOT / "outputs" / "reports" / "stage3_auditor_summary.md"


def main() -> None:
    df = pd.read_csv(IN_PATH)

    total = len(df)
    flagged = int((df["audit_decision"] == "manual_review").sum())
    accepted = int((df["audit_decision"] == "accept_model_output").sum())

    flagged_rate = flagged / total
    accepted_rate = accepted / total

    error_summary = (
        df.groupby("audit_decision")["is_error"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    risk_summary = df["risk_band"].value_counts().reset_index()
    risk_summary.columns = ["risk_band", "count"]
    risk_summary["percent"] = risk_summary["count"] / total

    reason_summary = df["audit_reasons"].value_counts().reset_index()
    reason_summary.columns = ["audit_reasons", "count"]
    reason_summary["percent"] = reason_summary["count"] / total

    false_negatives = df[
        (df["label"].astype(int) == 1)
        & (df["prediction_threshold_0_85"].astype(int) == 0)
    ]

    false_positives = df[
        (df["label"].astype(int) == 0)
        & (df["prediction_threshold_0_85"].astype(int) == 1)
    ]

    flagged_fn = int((false_negatives["audit_decision"] == "manual_review").sum())
    flagged_fp = int((false_positives["audit_decision"] == "manual_review").sum())

    lines = []
    lines.append("# Stage 3 Auditor Layer Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "Stage 3 adds a transparent rule-based auditor on top of the multimodal "
        "EfficientNet-B0 + metadata model. The auditor does not retrain the model. "
        "It reviews the model probability, selected threshold, image quality, artifact "
        "level, and borderline-risk behavior to decide whether the model output can be "
        "accepted or should be sent for manual review.\n"
    )

    lines.append("## Inputs\n")
    lines.append("- Model predictions: `outputs/reports/multimodal_test_predictions.csv`\n")
    lines.append("- Metadata test split: `data/processed/dr_multimodal_test.csv`\n")
    lines.append("- Output file: `outputs/reports/audited_multimodal_test_predictions.csv`\n\n")

    lines.append("## Auditor rules\n")
    lines.append("- Flag predictions near the selected threshold of 0.85.\n")
    lines.append("- Flag model-positive high-risk cases.\n")
    lines.append("- Flag model-negative cases with elevated probability >= 0.50.\n")
    lines.append("- Flag cases with inadequate image quality.\n")
    lines.append("- Flag rare high artifact level cases where `artifacts >= 2`.\n\n")

    lines.append("## Overall audit result\n")
    lines.append(f"- Total test cases: {total}\n")
    lines.append(f"- Accepted model outputs: {accepted} ({accepted_rate:.2%})\n")
    lines.append(f"- Sent to manual review: {flagged} ({flagged_rate:.2%})\n\n")

    lines.append("## Error concentration by audit decision\n\n")
    lines.append(error_summary.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Risk-band distribution\n\n")
    lines.append(risk_summary.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Main audit reasons\n\n")
    lines.append(reason_summary.head(15).to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Error capture\n")
    lines.append(f"- False negatives in test set: {len(false_negatives)}\n")
    lines.append(f"- False negatives sent to manual review: {flagged_fn}\n")
    lines.append(f"- False positives in test set: {len(false_positives)}\n")
    lines.append(f"- False positives sent to manual review: {flagged_fp}\n\n")

    lines.append("## Interpretation\n")
    lines.append(
        "The auditor creates a practical safety layer by separating low-risk accepted "
        "outputs from cases that deserve manual review. The accepted group has a much "
        "lower observed error rate, while the manual-review group contains a much higher "
        "share of errors and uncertainty. This supports the project goal of a "
        "self-correcting fundus screening copilot rather than a raw classifier.\n"
    )

    OUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
