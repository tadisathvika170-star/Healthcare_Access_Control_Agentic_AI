"""
preprocess.py

Preprocess DSL-StrongPasswordData for
3-repetition behavioral biometric authentication.

Input:
    dataset/DSL-StrongPasswordData.csv

Output:
    dataset/X_train.npy
    dataset/y_train.npy
    dataset/X_test.npy
    dataset/y_test.npy
    models/scaler.pkl
    config/label_encoder.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "dataset"
    / "DSL-StrongPasswordData.csv"
)

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"


# ==========================================================
# CONFIGURATION
# ==========================================================

SEQUENCE_LENGTH = 3

ID_COLUMNS = [
    "subject",
    "sessionIndex",
    "rep",
]


BEHAVIORAL_FEATURES = [
    "H.period",
    "DD.period.t",
    "UD.period.t",
    "H.t",
    "DD.t.i",
    "UD.t.i",
    "H.i",
    "DD.i.e",
    "UD.i.e",
    "H.e",
    "DD.e.five",
    "UD.e.five",
    "H.five",
    "DD.five.Shift.r",
    "UD.five.Shift.r",
    "H.Shift.r",
    "DD.Shift.r.o",
    "UD.Shift.r.o",
    "H.o",
    "DD.o.a",
    "UD.o.a",
    "H.a",
    "DD.a.n",
    "UD.a.n",
    "H.n",
    "DD.n.l",
    "UD.n.l",
    "H.l",
    "DD.l.Return",
    "UD.l.Return",
    "H.Return",
]


# ==========================================================
# DIRECTORIES
# ==========================================================

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("3-REPETITION DATA PREPROCESSING")
print("=" * 70)

print(f"\nDataset: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print(f"Original shape: {df.shape}")


# ==========================================================
# CHECK FEATURES
# ==========================================================

missing_features = [
    feature
    for feature in BEHAVIORAL_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing behavioral features:\n"
        + "\n".join(missing_features)
    )


print(
    f"Behavioral features: "
    f"{len(BEHAVIORAL_FEATURES)}"
)


# ==========================================================
# REMOVE MISSING VALUES
# ==========================================================

df = df.dropna(
    subset=BEHAVIORAL_FEATURES
).reset_index(drop=True)

print(
    f"After removing missing values: "
    f"{df.shape}"
)


# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

df["subject_encoded"] = label_encoder.fit_transform(
    df["subject"]
)

num_classes = len(
    label_encoder.classes_
)

print(
    f"Number of users/classes: "
    f"{num_classes}"
)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================
#
# Sessions 1-6 -> training
# Sessions 7-8 -> testing
#
# This preserves the original project split.
# ==========================================================

train_df = df[
    df["sessionIndex"].between(1, 6)
].copy()

test_df = df[
    df["sessionIndex"].between(7, 8)
].copy()


print(
    f"\nTraining rows: {len(train_df)}"
)

print(
    f"Testing rows : {len(test_df)}"
)


# ==========================================================
# SCALER
# ==========================================================

scaler = StandardScaler()

train_df[
    BEHAVIORAL_FEATURES
] = scaler.fit_transform(
    train_df[BEHAVIORAL_FEATURES]
)

test_df[
    BEHAVIORAL_FEATURES
] = scaler.transform(
    test_df[BEHAVIORAL_FEATURES]
)


# ==========================================================
# SAVE SCALER
# ==========================================================

scaler_path = MODEL_DIR / "scaler.pkl"

joblib.dump(
    scaler,
    scaler_path
)

print(
    f"\nScaler saved to: {scaler_path}"
)


# ==========================================================
# SAVE LABEL ENCODER
# ==========================================================

encoder_path = CONFIG_DIR / "label_encoder.pkl"

joblib.dump(
    label_encoder,
    encoder_path
)

print(
    f"Label encoder saved to: {encoder_path}"
)


# ==========================================================
# SEQUENCE CREATION
# ==========================================================

def create_sequences(dataframe):

    sequences = []
    labels = []

    # Group by user and session so
    # sequences never cross sessions/users.

    grouped = dataframe.groupby(
        ["subject", "sessionIndex"],
        sort=True
    )

    for (
        subject,
        session_index
    ), group in grouped:

        group = group.sort_values(
            "rep"
        )

        values = group[
            BEHAVIORAL_FEATURES
        ].to_numpy(
            dtype=np.float32
        )

        labels_for_group = group[
            "subject_encoded"
        ].to_numpy(
            dtype=np.int64
        )


        # Non-overlapping sequences.

        for start in range(
            0,
            len(values) - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH
        ):

            end = (
                start +
                SEQUENCE_LENGTH
            )

            sequence = values[
                start:end
            ]

            label_window = labels_for_group[
                start:end
            ]


            # Every step must belong
            # to the same user.

            if len(
                np.unique(label_window)
            ) != 1:

                continue


            sequences.append(
                sequence
            )

            labels.append(
                label_window[0]
            )


    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            labels,
            dtype=np.int64
        )
    )


# ==========================================================
# CREATE TRAIN SEQUENCES
# ==========================================================

X_train, y_train = create_sequences(
    train_df
)


# ==========================================================
# CREATE TEST SEQUENCES
# ==========================================================

X_test, y_test = create_sequences(
    test_df
)


# ==========================================================
# VALIDATE
# ==========================================================

expected_feature_count = 31

if X_train.ndim != 3:
    raise ValueError(
        f"Invalid X_train dimensions: "
        f"{X_train.shape}"
    )

if X_train.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(
        f"Expected sequence length "
        f"{SEQUENCE_LENGTH}, got "
        f"{X_train.shape[1]}"
    )

if X_train.shape[2] != expected_feature_count:
    raise ValueError(
        f"Expected {expected_feature_count} "
        f"features, got "
        f"{X_train.shape[2]}"
    )


# ==========================================================
# SAVE ARRAYS
# ==========================================================

np.save(
    DATASET_DIR / "X_train.npy",
    X_train
)

np.save(
    DATASET_DIR / "y_train.npy",
    y_train
)

np.save(
    DATASET_DIR / "X_test.npy",
    X_test
)

np.save(
    DATASET_DIR / "y_test.npy",
    y_test
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(
    f"Sequence length : "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Features        : "
    f"{X_train.shape[2]}"
)

print(
    f"Classes         : "
    f"{num_classes}"
)

print(
    f"\nX_train shape   : "
    f"{X_train.shape}"
)

print(
    f"y_train shape   : "
    f"{y_train.shape}"
)

print(
    f"X_test shape    : "
    f"{X_test.shape}"
)

print(
    f"y_test shape    : "
    f"{y_test.shape}"
)

print("\nSaved files:")

print(
    DATASET_DIR / "X_train.npy"
)

print(
    DATASET_DIR / "y_train.npy"
)

print(
    DATASET_DIR / "X_test.npy"
)

print(
    DATASET_DIR / "y_test.npy"
)

print(
    MODEL_DIR / "scaler.pkl"
)

print(
    CONFIG_DIR / "label_encoder.pkl"
)

print("=" * 70)