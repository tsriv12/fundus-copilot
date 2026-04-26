from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.multimodal_dataset import METADATA_COLUMNS
from src.models.multimodal_model import FundusMultimodalModel


CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "multimodal_efficientnet_b0_5epochs.pt"
TEST_CSV = ROOT / "data" / "processed" / "dr_multimodal_test.csv"

OUT_DIR = ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "metadata_input_shap_summary.csv"
OUT_MD = OUT_DIR / "metadata_input_shap_summary.md"

BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300
RANDOM_SEED = 42


class MetadataBranchWrapper(torch.nn.Module):
    """
    Wrapper for metadata branch only.

    This explains raw metadata inputs through the trained metadata MLP encoder.
    It summarizes which metadata inputs most affect the metadata representation.
    """

    def __init__(self, metadata_encoder: torch.nn.Module):
        super().__init__()
        self.metadata_encoder = metadata_encoder

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        # Return one scalar so SHAP can explain the metadata branch as a compact output.
        encoded = self.metadata_encoder(metadata)
        return encoded.mean(dim=1, keepdim=True)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(TEST_CSV)
    metadata = df[METADATA_COLUMNS].astype("float32").to_numpy()

    n_total = len(metadata)
    n_needed = min(n_total, BACKGROUND_SIZE + EXPLAIN_SIZE)

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(n_total, size=n_needed, replace=False)

    selected = metadata[selected_indices]
    background = torch.tensor(selected[:BACKGROUND_SIZE], dtype=torch.float32).to(device)
    explain_data = torch.tensor(
        selected[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE],
        dtype=torch.float32,
    ).to(device)

    model = FundusMultimodalModel(metadata_dim=len(METADATA_COLUMNS)).to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    wrapper = MetadataBranchWrapper(model.metadata_encoder).to(device)
    wrapper.eval()

    print("Running SHAP DeepExplainer on raw metadata inputs...")
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame({
        "metadata_feature": METADATA_COLUMNS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    total = result["mean_abs_shap"].sum()
    result["relative_importance"] = result["mean_abs_shap"] / total if total > 0 else 0

    result.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# Metadata Input SHAP Summary\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This analysis calculates SHAP values for the raw metadata inputs passing through "
        "the trained metadata MLP encoder. Mean absolute SHAP values are averaged across "
        "the explained test subset to rank metadata inputs by contribution to the metadata representation.\n\n"
    )

    lines.append("## Configuration\n\n")
    lines.append(f"- Background samples: {BACKGROUND_SIZE}\n")
    lines.append(f"- Explained test samples: {EXPLAIN_SIZE}\n")
    lines.append(f"- Metadata features: {len(METADATA_COLUMNS)}\n\n")

    lines.append("## Average absolute SHAP by metadata feature\n\n")
    lines.append(result.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Interpretation\n\n")
    lines.append(
        "Higher mean absolute SHAP values indicate metadata inputs that more strongly affect "
        "the trained metadata MLP branch. This complements the branch-level SHAP analysis, "
        "which showed the relative contribution of EfficientNet-B0 latent image features and "
        "metadata MLP latent features to the final fused classifier.\n"
    )

    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(f"Saved metadata SHAP CSV: {OUT_CSV}")
    print(f"Saved metadata SHAP Markdown: {OUT_MD}")
    print("\nMetadata SHAP result:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
