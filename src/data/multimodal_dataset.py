from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


METADATA_COLUMNS = [
    "patient_age",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "artifacts",
]


def build_transforms(image_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class FundusMultimodalDataset(Dataset):
    def __init__(self, csv_path, transform=None, metadata_cols=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.metadata_cols = metadata_cols or METADATA_COLUMNS

        missing_cols = [c for c in self.metadata_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing metadata columns: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        metadata = row[self.metadata_cols].to_numpy(dtype=np.float32)
        metadata = torch.tensor(metadata, dtype=torch.float32)

        label = torch.tensor(float(row["diabetic_retinopathy"]), dtype=torch.float32)

        return {
            "image": image,
            "metadata": metadata,
            "label": label,
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
        }

