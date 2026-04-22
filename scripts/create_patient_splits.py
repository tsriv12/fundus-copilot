from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "projects" / "radiology-copilot"
IN_PATH = ROOT / "data" / "processed" / "dr_binary_dataset.csv"
OUT_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42

df = pd.read_csv(IN_PATH)

# patient-level label for stratification:
# if any image for a patient is positive, mark patient positive
patient_df = (
    df.groupby("patient_id", as_index=False)["diabetic_retinopathy"]
      .max()
      .rename(columns={"diabetic_retinopathy": "patient_target"})
)

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=patient_df["patient_target"]
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_patients["patient_target"]
)

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])

train_df = df[df["patient_id"].isin(train_ids)].copy()
val_df = df[df["patient_id"].isin(val_ids)].copy()
test_df = df[df["patient_id"].isin(test_ids)].copy()

# sanity checks
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

split_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

split_df.to_csv(OUT_DIR / "dr_binary_splits.csv", index=False)
train_df.to_csv(OUT_DIR / "dr_binary_train.csv", index=False)
val_df.to_csv(OUT_DIR / "dr_binary_val.csv", index=False)
test_df.to_csv(OUT_DIR / "dr_binary_test.csv", index=False)

def summarize(name, frame):
    print(f"\n{name}")
    print(f"rows: {len(frame)}")
    print(f"patients: {frame['patient_id'].nunique()}")
    print("target counts:")
    print(frame["diabetic_retinopathy"].value_counts().sort_index())

print("Saved split files to:", OUT_DIR)
summarize("TRAIN", train_df)
summarize("VAL", val_df)
summarize("TEST", test_df)

print("\nLeakage check:")
print("train/val overlap:", len(set(train_df["patient_id"]) & set(val_df["patient_id"])))
print("train/test overlap:", len(set(train_df["patient_id"]) & set(test_df["patient_id"])))
print("val/test overlap:", len(set(val_df["patient_id"]) & set(test_df["patient_id"])))
