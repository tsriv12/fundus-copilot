import copy
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms
from src.models.multimodal_model import FundusMultimodalModel


def get_positive_class_weight(train_csv_path: Path) -> torch.Tensor:
    df = pd.read_csv(train_csv_path)
    pos = df["diabetic_retinopathy"].sum()
    neg = len(df) - pos
    pos_weight = neg / pos
    return torch.tensor([pos_weight], dtype=torch.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            logits = model(images, metadata)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"[val] batch {batch_idx}/{len(loader)}")

    return running_loss / total, correct / total


def main():
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = processed_dir / "dr_multimodal_train.csv"
    val_csv = processed_dir / "dr_multimodal_val.csv"

    train_ds = FundusMultimodalDataset(
        csv_path=train_csv,
        transform=build_transforms(image_size=224, train=True),
    )
    val_ds = FundusMultimodalDataset(
        csv_path=val_csv,
        transform=build_transforms(image_size=224, train=False),
    )

    batch_size = 32
    num_workers = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("batch_size:", batch_size)
    print("num_workers:", num_workers)
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))

    model = FundusMultimodalModel(metadata_dim=10).to(device)

    pos_weight = get_positive_class_weight(train_csv).to(device)
    print("pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        start = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images, metadata)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                print(f"[train] epoch {epoch + 1} batch {batch_idx}/{len(train_loader)}")

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    best_path = output_dir / "multimodal_efficientnet_b0_5epochs.pt"
    torch.save(best_state, best_path)
    print("saved best multimodal model to:", best_path)


if __name__ == "__main__":
    main()
