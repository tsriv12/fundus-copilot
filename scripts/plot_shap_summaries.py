from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BRANCH_CSV = ROOT / "outputs" / "reports" / "branch_shap_summary.csv"
METADATA_CSV = ROOT / "outputs" / "reports" / "metadata_input_shap_summary.csv"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_branch_shap() -> None:
    df = pd.read_csv(BRANCH_CSV)

    plt.figure(figsize=(8, 5))
    plt.bar(df["branch"], df["relative_total_importance"])
    plt.title("Branch-Level Relative SHAP Importance")
    plt.xlabel("Model Branch")
    plt.ylabel("Relative Total Importance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "branch_level_shap_importance.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_metadata_shap() -> None:
    df = pd.read_csv(METADATA_CSV).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(df["metadata_feature"], df["mean_abs_shap"])
    plt.title("Metadata Input Mean Absolute SHAP")
    plt.xlabel("Mean Absolute SHAP")
    plt.ylabel("Metadata Feature")
    plt.tight_layout()

    out = FIG_DIR / "metadata_input_shap_importance.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    plot_branch_shap()
    plot_metadata_shap()


if __name__ == "__main__":
    main()

