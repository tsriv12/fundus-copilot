from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms
from src.models.multimodal_model import FundusMultimodalModel


CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "multimodal_efficientnet_b0_5epochs.pt"
TEST_CSV = ROOT / "data" / "processed" / "dr_multimodal_test.csv"

OUT_DIR = ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURE_CSV = OUT_DIR / "branch_shap_feature_values.csv"
OUT_SUMMARY_CSV = OUT_DIR / "branch_shap_summary.csv"
OUT_MD = OUT_DIR / "branch_shap_summary.md"

BATCH_SIZE = 16
BACKGROUND_SIZE = 64
EXPLAIN_SIZE = 128
RANDOM_SEED = 42


class FusionClassifierWrapper(torch.nn.Module):
    """
    Wrapper around the trained fusion classifier.

    Input is the concatenated latent feature vector:
    [EfficientNet-B0 image features, metadata MLP features]
    """

    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_features)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


@torch.no_grad()
def extract_branch_features(model, loader, device):
    image_features_all = []
    metadata_features_all = []
    labels_all = []

    model.eval()

    for batch in loader:
        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].cpu().numpy()

        image_features = model.image_encoder(images)
        metadata_features = model.metadata_encoder(metadata)

        image_features_all.append(image_features.cpu())
        metadata_features_all.append(metadata_features.cpu())
        labels_all.append(labels)

    image_features_all = torch.cat(image_features_all, dim=0)
    metadata_features_all = torch.cat(metadata_features_all, dim=0)
    labels_all = np.concatenate(labels_all)

    return image_features_all, metadata_features_all, labels_all


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FundusMultimodalDataset(
        csv_path=TEST_CSV,
        transform=build_transforms(train=False),
    )

    n_total = len(dataset)
    n_needed = min(n_total, BACKGROUND_SIZE + EXPLAIN_SIZE)

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(n_total, size=n_needed, replace=False).tolist()

    selected_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(
        selected_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = FundusMultimodalModel(metadata_dim=10).to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    print("Extracting EfficientNet-B0 and MLP latent features...")
    image_features, metadata_features, labels = extract_branch_features(model, loader, device)

    fused_features = torch.cat([image_features, metadata_features], dim=1)

    image_dim = image_features.shape[1]
    metadata_dim = metadata_features.shape[1]
    fused_dim = fused_features.shape[1]

    print("Image feature dim:", image_dim)
    print("Metadata MLP feature dim:", metadata_dim)
    print("Fused feature dim:", fused_dim)

    background = fused_features[:BACKGROUND_SIZE].to(device)
    explain_data = fused_features[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE].to(device)

    wrapper = FusionClassifierWrapper(model.classifier).to(device)
    wrapper.eval()

    print("Running SHAP DeepExplainer on fused latent features...")
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    abs_shap = np.abs(shap_values)
    mean_abs_shap = abs_shap.mean(axis=0)

    feature_rows = []

    for i in range(image_dim):
        feature_rows.append({
            "branch": "efficientnet_b0_image_encoder",
            "feature": f"efficientnet_feature_{i}",
            "mean_abs_shap": float(mean_abs_shap[i]),
        })

    for j in range(metadata_dim):
        idx = image_dim + j
        feature_rows.append({
            "branch": "metadata_mlp_encoder",
            "feature": f"metadata_mlp_feature_{j}",
            "mean_abs_shap": float(mean_abs_shap[idx]),
        })

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(OUT_FEATURE_CSV, index=False)

    summary_df = (
        feature_df.groupby("branch")["mean_abs_shap"]
        .agg(["count", "mean", "sum", "max"])
        .reset_index()
        .rename(columns={
            "count": "num_features",
            "mean": "avg_mean_abs_shap_per_feature",
            "sum": "total_mean_abs_shap",
            "max": "max_single_feature_mean_abs_shap",
        })
    )

    total_importance = summary_df["total_mean_abs_shap"].sum()
    summary_df["relative_total_importance"] = (
        summary_df["total_mean_abs_shap"] / total_importance
        if total_importance > 0
        else 0
    )

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    top_image = (
        feature_df[feature_df["branch"] == "efficientnet_b0_image_encoder"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )

    top_metadata = (
        feature_df[feature_df["branch"] == "metadata_mlp_encoder"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )

    lines = []
    lines.append("# Branch-Level SHAP Summary\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This analysis calculates SHAP values on the fused latent feature space of the trained "
        "multimodal model. EfficientNet-B0 image features and metadata-MLP features are extracted "
        "from the trained model, then SHAP is applied to the final fusion classifier. Mean absolute "
        "SHAP values are averaged across the explained test subset.\n\n"
    )

    lines.append("## Configuration\n\n")
    lines.append(f"- Background samples: {BACKGROUND_SIZE}\n")
    lines.append(f"- Explained test samples: {EXPLAIN_SIZE}\n")
    lines.append(f"- EfficientNet-B0 latent image features: {image_dim}\n")
    lines.append(f"- Metadata MLP latent features: {metadata_dim}\n\n")

    lines.append("## Branch-level average SHAP result\n\n")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Top EfficientNet-B0 latent features by mean absolute SHAP\n\n")
    lines.append(top_image.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Top metadata MLP latent features by mean absolute SHAP\n\n")
    lines.append(top_metadata.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Interpretation\n\n")
    lines.append(
        "The table reports average absolute SHAP values for the EfficientNet-B0 image branch and "
        "the metadata MLP branch. The total mean absolute SHAP value gives a branch-level estimate "
        "of how much each branch contributes to the final fused classifier over the explained test subset. "
        "This is a latent-feature explanation rather than a pixel-level medical heatmap.\n"
    )

    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(f"Saved feature SHAP values: {OUT_FEATURE_CSV}")
    print(f"Saved branch SHAP summary: {OUT_SUMMARY_CSV}")
    print(f"Saved Markdown summary: {OUT_MD}")
    print("\nBranch summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

