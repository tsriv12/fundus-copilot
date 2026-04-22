from pathlib import Path
import pandas as pd

ROOT = Path.home() / "projects" / "radiology-copilot"
DATASET_DIR = ROOT / "data" / "raw" / "a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1"
CSV_PATH = DATASET_DIR / "labels_brset.csv"
IMG_DIR = DATASET_DIR / "fundus_photos"
OUT_PATH = ROOT / "data" / "processed" / "dr_binary_dataset.csv"

df = pd.read_csv(CSV_PATH).copy()

df["image_file"] = df["image_id"].astype(str).str.strip() + ".jpg"
df["image_path"] = df["image_file"].apply(lambda x: str(IMG_DIR / x))

# keep only rows whose image actually exists
df = df[df["image_path"].apply(lambda x: Path(x).exists())].copy()

# keep only the fields we need for the first baseline
keep_cols = [
    "image_id",
    "image_file",
    "image_path",
    "patient_id",
    "patient_age",
    "patient_sex",
    "exam_eye",
    "camera",
    "diabetic_retinopathy",
    "quality",
]
df = df[keep_cols].copy()

# force target to integer
df["diabetic_retinopathy"] = pd.to_numeric(df["diabetic_retinopathy"], errors="coerce")
df = df.dropna(subset=["diabetic_retinopathy"]).copy()
df["diabetic_retinopathy"] = df["diabetic_retinopathy"].astype(int)

df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df["diabetic_retinopathy"].value_counts(dropna=False).sort_index())

print("\nQuality distribution:")
print(df["quality"].value_counts(dropna=False))

print("\nSample rows:")
print(df.head())
