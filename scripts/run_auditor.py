from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agents.auditor import audit_prediction


PRED_PATH = ROOT / "outputs" / "reports" / "multimodal_test_predictions.csv"
TEST_META_PATH = ROOT / "data" / "processed" / "dr_multimodal_test.csv"
OUT_PATH = ROOT / "outputs" / "reports" / "audited_multimodal_test_predictions.csv"


def main() -> None:
    preds = pd.read_csv(PRED_PATH)
    meta = pd.read_csv(TEST_META_PATH)

    print("Predictions shape:", preds.shape)
    print("Metadata test shape:", meta.shape)

    # Keep useful metadata columns only if they exist.
    candidate_meta_cols = [
        "image_id",
        "patient_id",
        "quality",
        "quality_flag",
        "artifacts",
        "patient_age",
        "patient_sex",
        "exam_eye",
        "diabetes",
    ]
    meta_cols = [c for c in candidate_meta_cols if c in meta.columns]

    merged = preds.merge(
        meta[meta_cols],
        on=["image_id", "patient_id"],
        how="left",
        validate="one_to_one",
    )

    print("Merged shape:", merged.shape)
    print("Merged columns:", merged.columns.tolist())

    audit_rows = []
    for _, row in merged.iterrows():
        audit = audit_prediction(
            probability=float(row["probability"]),
            prediction=int(row["prediction_threshold_0_85"]),
            threshold=0.85,
            quality=row["quality"] if "quality" in merged.columns else None,
            artifacts=row["artifacts"] if "artifacts" in merged.columns else None,
            margin=0.10,
        )
        audit_rows.append(audit)

    audit_df = pd.DataFrame(audit_rows)
    final = pd.concat([merged, audit_df], axis=1)

    final["is_error"] = (
        final["label"].astype(int) != final["prediction_threshold_0_85"].astype(int)
    ).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("\nAudit decision counts:")
    print(final["audit_decision"].value_counts())

    print("\nRisk band counts:")
    print(final["risk_band"].value_counts())

    print("\nError rate by audit decision:")
    print(final.groupby("audit_decision")["is_error"].agg(["count", "sum", "mean"]))

    print("\nTop audit reasons:")
    print(final["audit_reasons"].value_counts().head(20))


if __name__ == "__main__":
    main()

