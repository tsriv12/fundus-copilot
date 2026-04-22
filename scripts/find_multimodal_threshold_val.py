import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms
from src.models.multimodal_model import FundusMultimodalModel


def main():
    val_csv = project_root / "data" / "processed" / "dr_multimodal_val.csv"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "multimodal_efficientnet_b0_5epochs.pt"

    val_ds = FundusMultimodalDataset(
        csv_path=val_csv,
        transform=build_transforms(image_size=224, train=False),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print("val size:", len(val_ds))
    print("checkpoint:", checkpoint_path)

    model = FundusMultimodalModel(metadata_dim=10).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)

            logits = model(images, metadata)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            y_true.extend(batch["label"].cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

            if batch_idx % 25 == 0 or batch_idx == len(val_loader):
                print(f"[val] batch {batch_idx}/{len(val_loader)}")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    best_f1 = -1
    best_threshold = None
    best_metrics = None

    print("\n=== Validation Threshold Sweep ===")

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        acc_t = accuracy_score(y_true, y_pred_t)
        prec_t = precision_score(y_true, y_pred_t, zero_division=0)
        rec_t = recall_score(y_true, y_pred_t, zero_division=0)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        cm_t = confusion_matrix(y_true, y_pred_t)

        print(f"\n--- threshold={t:.2f} ---")
        print(f"accuracy:  {acc_t:.4f}")
        print(f"precision: {prec_t:.4f}")
        print(f"recall:    {rec_t:.4f}")
        print(f"f1-score:  {f1_t:.4f}")
        print("confusion_matrix:")
        print(cm_t)

        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t
            best_metrics = {
                "accuracy": acc_t,
                "precision": prec_t,
                "recall": rec_t,
                "f1": f1_t,
                "confusion_matrix": cm_t,
            }

    print("\n=== Best Validation Threshold by F1 ===")
    print(f"best_threshold: {best_threshold:.2f}")
    print(f"accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"precision: {best_metrics['precision']:.4f}")
    print(f"recall:    {best_metrics['recall']:.4f}")
    print(f"f1-score:  {best_metrics['f1']:.4f}")
    print("confusion_matrix:")
    print(best_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
