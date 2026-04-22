from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[1]

input_csv = project_root / "data" / "processed" / "dr_multimodal_dataset.csv"
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv)

target_col = "diabetic_retinopathy"
patient_col = "patient_id"

patient_df = (
    df.groupby(patient_col)[target_col]
    .max()
    .reset_index()
)

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    stratify=patient_df[target_col],
    random_state=42,
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    stratify=temp_patients[target_col],
    random_state=42,
)

train_ids = set(train_patients[patient_col])
val_ids = set(val_patients[patient_col])
test_ids = set(test_patients[patient_col])

train_df = df[df[patient_col].isin(train_ids)].copy()
val_df = df[df[patient_col].isin(val_ids)].copy()
test_df = df[df[patient_col].isin(test_ids)].copy()

train_path = output_dir / "dr_multimodal_train.csv"
val_path = output_dir / "dr_multimodal_val.csv"
test_path = output_dir / "dr_multimodal_test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("Saved split files to:", output_dir)

for name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    print(f"\n{name}")
    print("rows:", len(split_df))
    print("patients:", split_df[patient_col].nunique())
    print("target counts:")
    print(split_df[target_col].value_counts().sort_index())

print("\nLEAKAGE CHECK")
print("train-val overlap:", len(train_ids & val_ids))
print("train-test overlap:", len(train_ids & test_ids))
print("val-test overlap:", len(val_ids & test_ids))

