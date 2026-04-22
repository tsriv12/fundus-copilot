import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms
from src.models.multimodal_model import FundusMultimodalModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/multimodal_efficientnet_b0_5epochs.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reports/multimodal_test_predictions.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_csv = project_root / "data" / "processed" / "dr_multimodal_test.csv"
    checkpoint_path = project_root / args.checkpoint
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_ds = FundusMultimodalDataset(
        csv_path=test_csv,
        transform=build_transforms(image_size=224, train=False),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print("test size:", len(test_ds))
    print("checkpoint:", checkpoint_path)

    model = FundusMultimodalModel(metadata_dim=10).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_prob = []
    image_ids = []
    patient_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)

            logits = model(images, metadata)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            y_true.extend(batch["label"].cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())
            image_ids.extend(batch["image_id"])
            patient_ids.extend(batch["patient_id"].cpu().numpy().tolist() if torch.is_tensor(batch["patient_id"]) else list(batch["patient_id"]))

            if batch_idx % 25 == 0 or batch_idx == len(test_loader):
                print(f"[test] batch {batch_idx}/{len(test_loader)}")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    
    best_f1 = -1
    best_threshold = None
    best_metrics = None

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        acc_t = accuracy_score(y_true, y_pred_t)
        prec_t = precision_score(y_true, y_pred_t, zero_division=0)
        rec_t = recall_score(y_true, y_pred_t, zero_division=0)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        cm_t = confusion_matrix(y_true, y_pred_t)

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

    print("\n=== Best Threshold by F1 ===")
    print(f"best_threshold: {best_threshold:.2f}")
    print(f"accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"precision: {best_metrics['precision']:.4f}")
    print(f"recall:    {best_metrics['recall']:.4f}")
    print(f"f1-score:  {best_metrics['f1']:.4f}")
    print("confusion_matrix:")
    print(best_metrics["confusion_matrix"])

    print("\n=== Threshold Sweep ===")
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

    selected_threshold = 0.85
    y_pred = (y_prob >= selected_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Test Metrics ===")
    print(f"selected_threshold: {selected_threshold:.2f}")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1-score:  {f1:.4f}")
    print(f"roc_auc:   {roc_auc:.4f}")
    print(f"pr_auc:    {pr_auc:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    out_df = pd.DataFrame({
        "image_id": image_ids,
        "patient_id": patient_ids,
        "label": y_true,
        "probability": y_prob,
        "prediction_threshold_0_85": y_pred,
    })

    out_df.to_csv(output_path, index=False)
    print("saved predictions to:", output_path)


if __name__ == "__main__":
    main()
