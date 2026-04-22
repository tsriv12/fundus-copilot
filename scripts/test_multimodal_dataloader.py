import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms


def main():
    processed_dir = project_root / "data" / "processed"

    train_csv = processed_dir / "dr_multimodal_train.csv"
    val_csv = processed_dir / "dr_multimodal_val.csv"
    test_csv = processed_dir / "dr_multimodal_test.csv"

    train_ds = FundusMultimodalDataset(
        csv_path=train_csv,
        transform=build_transforms(image_size=224, train=True),
    )
    val_ds = FundusMultimodalDataset(
        csv_path=val_csv,
        transform=build_transforms(image_size=224, train=False),
    )
    test_ds = FundusMultimodalDataset(
        csv_path=test_csv,
        transform=build_transforms(image_size=224, train=False),
    )

    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("test size:", len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(train_loader))

    print("image shape:", batch["image"].shape)
    print("metadata shape:", batch["metadata"].shape)
    print("label shape:", batch["label"].shape)
    print("metadata dtype:", batch["metadata"].dtype)
    print("label dtype:", batch["label"].dtype)
    print("first metadata row:", batch["metadata"][0])
    print("first 5 labels:", batch["label"][:5])


if __name__ == "__main__":
    main()

