"""Preprocess DSL-StrongPasswordData for single-repetition authentication."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "DSL-StrongPasswordData.csv"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

SEQUENCE_LENGTH = 1
ID_COLUMNS = ["subject", "sessionIndex", "rep"]
BEHAVIORAL_FEATURES = [
    "H.period", "DD.period.t", "UD.period.t", "H.t", "DD.t.i", "UD.t.i",
    "H.i", "DD.i.e", "UD.i.e", "H.e", "DD.e.five", "UD.e.five", "H.five",
    "DD.five.Shift.r", "UD.five.Shift.r", "H.Shift.r", "DD.Shift.r.o",
    "UD.Shift.r.o", "H.o", "DD.o.a", "UD.o.a", "H.a", "DD.a.n", "UD.a.n",
    "H.n", "DD.n.l", "UD.n.l", "H.l", "DD.l.Return", "UD.l.Return", "H.Return",
]

for directory in (DATASET_DIR, MODEL_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
missing = [f for f in BEHAVIORAL_FEATURES if f not in df.columns]
if missing:
    raise ValueError("Missing behavioral features: " + ", ".join(missing))
df = df.dropna(subset=BEHAVIORAL_FEATURES).reset_index(drop=True)

encoder = LabelEncoder()
df["subject_encoded"] = encoder.fit_transform(df["subject"])
train_df = df[df["sessionIndex"].between(1, 6)].copy()
test_df = df[df["sessionIndex"].between(7, 8)].copy()

scaler = StandardScaler()
train_df[BEHAVIORAL_FEATURES] = scaler.fit_transform(train_df[BEHAVIORAL_FEATURES])
test_df[BEHAVIORAL_FEATURES] = scaler.transform(test_df[BEHAVIORAL_FEATURES])
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(encoder, CONFIG_DIR / "label_encoder.pkl")

def create_sequences(dataframe):
    sequences, labels = [], []
    for (_, _), group in dataframe.groupby(["subject", "sessionIndex"], sort=True):
        group = group.sort_values("rep")
        values = group[BEHAVIORAL_FEATURES].to_numpy(dtype=np.float32)
        labels_window = group["subject_encoded"].to_numpy(dtype=np.int64)
        for start in range(0, len(values) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            end = start + SEQUENCE_LENGTH
            label_window = labels_window[start:end]
            if len(np.unique(label_window)) != 1:
                continue
            sequences.append(values[start:end])
            labels.append(label_window[0])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)

X_train, y_train = create_sequences(train_df)
X_test, y_test = create_sequences(test_df)

if X_train.ndim != 3 or X_train.shape[1:] != (SEQUENCE_LENGTH, len(BEHAVIORAL_FEATURES)):
    raise ValueError(f"Invalid X_train shape: {X_train.shape}")

np.save(DATASET_DIR / "X_train.npy", X_train)
np.save(DATASET_DIR / "y_train.npy", y_train)
np.save(DATASET_DIR / "X_test.npy", X_test)
np.save(DATASET_DIR / "y_test.npy", y_test)

print("=" * 70)
print("SINGLE-REPETITION PREPROCESSING COMPLETED")
print(f"Sequence length : {SEQUENCE_LENGTH}")
print(f"Features        : {len(BEHAVIORAL_FEATURES)}")
print(f"Classes         : {len(encoder.classes_)}")
print(f"X_train shape   : {X_train.shape}")
print(f"y_train shape   : {y_train.shape}")
print(f"X_test shape    : {X_test.shape}")
print(f"y_test shape    : {y_test.shape}")
print("=" * 70)
