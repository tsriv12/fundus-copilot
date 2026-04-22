from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
comparison_csv = project_root / "outputs" / "reports" / "model_comparison_summary.csv"
figures_dir = project_root / "outputs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(comparison_csv)

display_names = {
    "image_only": "Image-only",
    "multimodal": "Multimodal",
}

df["display_model"] = df["model"].map(display_names)

metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

metric_df = df[["display_model"] + metric_cols].set_index("display_model").T

plt.figure(figsize=(10, 6))
metric_df.plot(kind="bar")
plt.title("Stage 2 Model Metric Comparison")
plt.ylabel("Score")
plt.xlabel("Metric")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

metric_plot_path = figures_dir / "stage2_model_metric_comparison.png"
plt.savefig(metric_plot_path, dpi=200)
plt.close()

error_cols = ["false_positives", "false_negatives"]
error_df = df[["display_model"] + error_cols].set_index("display_model").T

plt.figure(figsize=(8, 5))
error_df.plot(kind="bar")
plt.title("Stage 2 Error Comparison")
plt.ylabel("Count")
plt.xlabel("Error Type")
plt.xticks(rotation=0)
plt.tight_layout()

error_plot_path = figures_dir / "stage2_error_comparison.png"
plt.savefig(error_plot_path, dpi=200)
plt.close()

print("saved:", metric_plot_path)
print("saved:", error_plot_path)
