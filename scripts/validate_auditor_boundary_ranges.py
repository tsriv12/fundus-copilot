from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "outputs" / "reports" / "audited_multimodal_test_predictions.csv"
OUT_CSV = ROOT / "outputs" / "reports" / "auditor_boundary_range_validation.csv"
OUT_MD = ROOT / "outputs" / "reports" / "auditor_boundary_range_validation.md"

THRESHOLD = 0.85
MARGINS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]


def evaluate_margin(df: pd.DataFrame, margin: float) -> dict:
    lower = THRESHOLD - margin
    upper = THRESHOLD + margin

    # Professor Option A:
    # Flag cases near decision boundary and validate against actual labels.
    flagged = df[(df["probability"] >= lower) & (df["probability"] <= upper)].copy()
    accepted = df.drop(flagged.index).copy()

    total = len(df)
    flagged_count = len(flagged)
    accepted_count = len(accepted)

    actual_dr_count_flagged = int((flagged["label"].astype(int) == 1).sum())
    actual_dr_pct_flagged = (
        actual_dr_count_flagged / flagged_count if flagged_count else 0.0
    )

    overall_dr_pct = float((df["label"].astype(int) == 1).mean())

    flagged_error_rate = float(flagged["is_error"].mean()) if flagged_count else 0.0
    accepted_error_rate = float(accepted["is_error"].mean()) if accepted_count else 0.0

    false_negatives = df[
        (df["label"].astype(int) == 1)
        & (df["prediction_threshold_0_85"].astype(int) == 0)
    ]

    false_positives = df[
        (df["label"].astype(int) == 0)
        & (df["prediction_threshold_0_85"].astype(int) == 1)
    ]

    flagged_fn = len(flagged.loc[flagged.index.intersection(false_negatives.index)])
    flagged_fp = len(flagged.loc[flagged.index.intersection(false_positives.index)])

    return {
        "threshold": THRESHOLD,
        "margin": margin,
        "lower_bound": lower,
        "upper_bound": upper,
        "total_cases": total,
        "flagged_cases": flagged_count,
        "flagged_pct": flagged_count / total,
        "accepted_cases": accepted_count,
        "accepted_pct": accepted_count / total,
        "actual_dr_count_in_flagged": actual_dr_count_flagged,
        "actual_dr_pct_in_flagged": actual_dr_pct_flagged,
        "overall_dr_pct": overall_dr_pct,
        "dr_enrichment_ratio": (
            actual_dr_pct_flagged / overall_dr_pct if overall_dr_pct else 0.0
        ),
        "flagged_error_rate": flagged_error_rate,
        "accepted_error_rate": accepted_error_rate,
        "false_negatives_flagged": flagged_fn,
        "false_negatives_total": len(false_negatives),
        "false_negatives_capture_pct": (
            flagged_fn / len(false_negatives) if len(false_negatives) else 0.0
        ),
        "false_positives_flagged": flagged_fp,
        "false_positives_total": len(false_positives),
        "false_positives_capture_pct": (
            flagged_fp / len(false_positives) if len(false_positives) else 0.0
        ),
    }


def main() -> None:
    df = pd.read_csv(IN_PATH)

    results = pd.DataFrame([evaluate_margin(df, m) for m in MARGINS])
    results.to_csv(OUT_CSV, index=False)

    best = results.sort_values(
        by=["dr_enrichment_ratio", "accepted_error_rate"],
        ascending=[False, True],
    ).iloc[0]

    lines = []
    lines.append("# Auditor Boundary Range Validation\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This analysis validates different near-threshold auditor ranges around "
        "the multimodal model decision threshold of 0.85. For each range, the "
        "script calculates what percentage of cases would be flagged for manual "
        "review and checks how many of those flagged cases actually have diabetic "
        "retinopathy in the test dataset.\n\n"
    )

    lines.append("## Boundary range results\n\n")
    lines.append(results.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Selected range\n\n")
    lines.append(
        f"The strongest DR enrichment was observed at margin ±{best['margin']:.2f}, "
        f"corresponding to probability range "
        f"[{best['lower_bound']:.2f}, {best['upper_bound']:.2f}].\n\n"
    )

    lines.append("## Interpretation\n\n")
    lines.append(
        "A useful auditor range should not simply flag many cases. It should flag "
        "a reviewable subset that is enriched for actual DR cases or model errors. "
        "This table supports choosing the decision-boundary range based on validation "
        "behavior rather than arbitrary manual selection.\n"
    )

    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved Markdown: {OUT_MD}")
    print("\nResults:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

