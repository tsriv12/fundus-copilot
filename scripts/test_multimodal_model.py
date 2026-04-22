import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.multimodal_dataset import FundusMultimodalDataset, build_transforms
from src.models.multimodal_model import FundusMultimodalModel


def main():
    processed_dir = project_root / "data" / "processed"
    train_csv = processed_dir / "dr_multimodal_train.csv"

    ds = FundusMultimodalDataset(
        csv_path=train_csv,
        transform=build_transforms(image_size=224, train=True),
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    images = batch["image"]
    metadata = batch["metadata"]

    model = FundusMultimodalModel(metadata_dim=metadata.shape[1])

    with torch.no_grad():
        logits = model(images, metadata)

    print("image shape:", images.shape)
    print("metadata shape:", metadata.shape)
    print("logits shape:", logits.shape)
    print("sample logits:", logits[:5].squeeze(1))


if __name__ == "__main__":
    main()

