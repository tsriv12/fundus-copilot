from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs" / "reports" / "audited_multimodal_test_predictions.csv"
FIG_DIR = ROOT / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_audit_decision_plot(df: pd.DataFrame) -> None:
    counts = df["audit_decision"].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Stage 3 Auditor Decision Distribution")
    plt.xlabel("Audit Decision")
    plt.ylabel("Number of Test Cases")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "stage3_audit_decision_distribution.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def save_error_rate_plot(df: pd.DataFrame) -> None:
    summary = df.groupby("audit_decision")["is_error"].mean().sort_values()

    plt.figure(figsize=(7, 5))
    summary.plot(kind="bar")
    plt.title("Observed Error Rate by Audit Decision")
    plt.xlabel("Audit Decision")
    plt.ylabel("Error Rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "stage3_error_rate_by_audit_decision.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def save_risk_band_plot(df: pd.DataFrame) -> None:
    counts = df["risk_band"].value_counts().reindex(["low", "medium", "high"])

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Risk Band Distribution")
    plt.xlabel("Risk Band")
    plt.ylabel("Number of Test Cases")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIG_DIR / "stage3_risk_band_distribution.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    df = pd.read_csv(IN_PATH)

    print("Loaded:", IN_PATH)
    print("Shape:", df.shape)

    save_audit_decision_plot(df)
    save_error_rate_plot(df)
    save_risk_band_plot(df)


if __name__ == "__main__":
    main()
