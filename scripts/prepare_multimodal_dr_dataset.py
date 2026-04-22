from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]

raw_csv = project_root / "data" / "raw" / "a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1" / "labels_brset.csv"
image_dir = project_root / "data" / "raw" / "a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1" / "fundus_photos"
output_csv = project_root / "data" / "processed" / "dr_multimodal_dataset.csv"

df = pd.read_csv(raw_csv)

keep_cols = [
    "image_id",
    "patient_id",
    "patient_age",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "artifacts",
    "quality",
    "diabetic_retinopathy",
]

df = df[keep_cols].copy()

df["image_path"] = df["image_id"].astype(str).apply(lambda x: str(image_dir / f"{x}.jpg"))

df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
age_median = df["patient_age"].median()
df["patient_age"] = df["patient_age"].fillna(age_median)

binary_cols = [
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "artifacts",
    "diabetic_retinopathy",
]

for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

df["patient_sex"] = pd.to_numeric(df["patient_sex"], errors="coerce").fillna(0).astype(int)
df["exam_eye"] = pd.to_numeric(df["exam_eye"], errors="coerce").fillna(0).astype(int)

df["quality"] = df["quality"].astype(str).str.strip().str.lower()

quality_map = {
    "adequate": 1,
    "inadequate": 0,
}

df["quality_flag"] = df["quality"].map(quality_map)

before_drop = len(df)
df = df.dropna(subset=["quality_flag"]).copy()
after_drop = len(df)

df["quality_flag"] = df["quality_flag"].astype(int)

ordered_cols = [
    "image_id",
    "image_path",
    "patient_id",
    "patient_age",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "artifacts",
    "quality",
    "quality_flag",
    "diabetic_retinopathy",
]

df = df[ordered_cols]
df.to_csv(output_csv, index=False)

print("saved to:", output_csv)
print("shape:", df.shape)
print("dropped rows due to unmapped quality:", before_drop - after_drop)
print("\nmissing values after processing:")
print(df.isna().sum())
print("\nhead:")
print(df.head(3).to_string())
print("\ntarget distribution:")
print(df["diabetic_retinopathy"].value_counts(dropna=False))
print("\nquality distribution:")
print(df["quality"].value_counts(dropna=False))
