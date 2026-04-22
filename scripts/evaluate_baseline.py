from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.data.dataset import FundusDRDataset, build_transforms


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def main():
    processed_dir = project_root / "data" / "processed"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "baseline_efficientnet_b0_debug.pt"

    test_csv = processed_dir / "dr_binary_test.csv"

    test_ds = FundusDRDataset(
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

    model = build_model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_prob = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()

            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            y_true.extend(labels.astype(int))
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())

            if batch_idx % 25 == 0 or batch_idx == len(test_loader):
                print(f"[test] batch {batch_idx}/{len(test_loader)}")
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

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


    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Test Metrics ===")
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

    results = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    })
    out_path = project_root / "outputs" / "reports" / "baseline_test_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print("\nsaved predictions to:", out_path)


if __name__ == "__main__":
    main()
