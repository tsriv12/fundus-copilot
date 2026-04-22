from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import FundusDRDataset, build_transforms


def main():
    processed_dir = project_root / "data" / "processed"

    train_csv = processed_dir / "dr_binary_train.csv"
    val_csv = processed_dir / "dr_binary_val.csv"
    test_csv = processed_dir / "dr_binary_test.csv"

    train_ds = FundusDRDataset(
        csv_path=train_csv,
        transform=build_transforms(image_size=224, train=True),
    )
    val_ds = FundusDRDataset(
        csv_path=val_csv,
        transform=build_transforms(image_size=224, train=False),
    )
    test_ds = FundusDRDataset(
        csv_path=test_csv,
        transform=build_transforms(image_size=224, train=False),
    )

    print(f"train size: {len(train_ds)}")
    print(f"val size: {len(val_ds)}")
    print(f"test size: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(train_loader))

    print("batch image shape:", batch["image"].shape)
    print("batch label shape:", batch["label"].shape)
    print("batch label dtype:", batch["label"].dtype)
    print("first 5 labels:", batch["label"][:5])


if __name__ == "__main__":
    main()
