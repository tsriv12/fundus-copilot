from pathlib import Path
import pandas as pd

DATASET_DIR = Path.home() / "projects" / "radiology-copilot" / "data" / "raw" / "a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1"
CSV_PATH = DATASET_DIR / "labels_brset.csv"
IMG_DIR = DATASET_DIR / "fundus_photos"

df = pd.read_csv(CSV_PATH)

print("CSV shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# Normalize image ids to filenames like img00001.jpg
df["image_file"] = df["image_id"].astype(str).str.strip() + ".jpg"

image_files = {p.name for p in IMG_DIR.glob("*.jpg")}
csv_files = set(df["image_file"])

missing_on_disk = sorted(csv_files - image_files)
missing_in_csv = sorted(image_files - csv_files)

print("\nUnique image_id count in CSV:", df["image_id"].nunique())
print("Image files on disk:", len(image_files))
print("CSV image references:", len(csv_files))

print("\nMissing on disk (referenced in CSV but file absent):", len(missing_on_disk))
print("Missing in CSV (file exists but not referenced):", len(missing_in_csv))

if missing_on_disk:
    print("\nSample missing on disk:", missing_on_disk[:10])

if missing_in_csv:
    print("\nSample missing in CSV:", missing_in_csv[:10])

print("\nNull counts (top 30):")
print(df.isna().sum().sort_values(ascending=False).head(30))
