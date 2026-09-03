"""
test_authentication.py

Complete authentication evaluation for the
Healthcare Access Control System.

Uses:
    - Existing trained LSTM model
    - Existing trained GRU model
    - Existing LSTM + GRU Fusion
    - Existing StandardScaler
    - Existing LabelEncoder

Calculates:
    - LSTM identification accuracy
    - GRU identification accuracy
    - Fusion identification accuracy
    - Genuine Acceptance
    - Genuine Rejection
    - False Acceptance Rate (FAR)
    - False Rejection Rate (FRR)
    - True Acceptance Rate (TAR)
    - True Rejection Rate (TRR)

IMPORTANT:
This file does NOT retrain or modify the models.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from config import (
    DATASET_PATH,
    LABEL_ENCODER_PATH,
)

from predict.predictor import Predictor


# ==========================================================
# CONFIGURATION
# ==========================================================

print("=" * 70)
print("HEALTHCARE ACCESS CONTROL")
print("LSTM + GRU AUTHENTICATION EVALUATION")
print("=" * 70)
print()


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

print("Loading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

usernames = list(
    label_encoder.classes_
)

print(
    f"Number of Users : {len(usernames)}"
)

print()


# ==========================================================
# LOAD RAW DATASET
# ==========================================================

print("Loading RAW dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset Shape : {df.shape}"
)

print()


# ==========================================================
# REMOVE MISSING VALUES
# ==========================================================

df = df.dropna().reset_index(
    drop=True
)

print(
    f"After Removing Missing Values : "
    f"{df.shape}"
)

print()


# ==========================================================
# IDENTIFY TARGET COLUMN
# ==========================================================

if "subject" in df.columns:

    target_column = "subject"

elif "user" in df.columns:

    target_column = "user"

elif "username" in df.columns:

    target_column = "username"

else:

    raise ValueError(
        "Could not find user/subject column "
        "in the dataset."
    )


print(
    f"Target Column : {target_column}"
)

print()


# ==========================================================
# FEATURES AND LABELS
# ==========================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# ==========================================================
# ENSURE CORRECT FEATURE ORDER
# ==========================================================

print(
    f"Number of Features : {X.shape[1]}"
)

if X.shape[1] != 33:

    raise ValueError(
        f"Expected 33 features, "
        f"but found {X.shape[1]}."
    )


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================
#
# Your existing preprocessing uses:
#
#     80% training
#     20% testing
#
# We reproduce the same split.
#
# random_state is taken from config if available.
#
# ==========================================================

try:

    from config import RANDOM_STATE

except ImportError:

    RANDOM_STATE = 42


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training Samples : {len(X_train)}"
)

print(
    f"Testing Samples  : {len(X_test)}"
)

print()


# ==========================================================
# LOAD FUSION PREDICTOR
# ==========================================================

print(
    "Loading LSTM + GRU Fusion Predictor..."
)

predictor = Predictor()

print(
    "Fusion Predictor loaded successfully."
)

print()


# ==========================================================
# STORAGE FOR PREDICTIONS
# ==========================================================

lstm_predictions = []

gru_predictions = []

fusion_predictions = []

actual_users = []


# ==========================================================
# EVALUATION
# ==========================================================

print(
    "Evaluating test samples..."
)

print()


for index in range(
    len(X_test)
):

    # ------------------------------------------------------
    # Get RAW features
    # ------------------------------------------------------

    sample = X_test.iloc[index]

    # Convert to NumPy array.
    #
    # DO NOT SCALE HERE.
    #
    # The LSTM and GRU predictors perform scaling.
    # ------------------------------------------------------

    features = sample.to_numpy(
        dtype=np.float64
    )

    # ------------------------------------------------------
    # Actual user
    # ------------------------------------------------------

    actual_user = str(
        y_test.iloc[index]
    )

    # ------------------------------------------------------
    # Fusion prediction
    # ------------------------------------------------------

    result = predictor.predict(
        features
    )

    # ------------------------------------------------------
    # Individual predictions
    # ------------------------------------------------------

    lstm_user = result[
        "lstm_prediction"
    ]

    gru_user = result[
        "gru_prediction"
    ]

    fusion_user = result[
        "predicted_user"
    ]

    # ------------------------------------------------------
    # Store results
    # ------------------------------------------------------

    actual_users.append(
        actual_user
    )

    lstm_predictions.append(
        lstm_user
    )

    gru_predictions.append(
        gru_user
    )

    fusion_predictions.append(
        fusion_user
    )


# ==========================================================
# IDENTIFICATION ACCURACY
# ==========================================================

lstm_accuracy = accuracy_score(
    actual_users,
    lstm_predictions
)

gru_accuracy = accuracy_score(
    actual_users,
    gru_predictions
)

fusion_accuracy = accuracy_score(
    actual_users,
    fusion_predictions
)


# ==========================================================
# GENUINE AUTHENTICATION
# ==========================================================
#
# Genuine attempt:
#
# Actual user == claimed user
#
# Your authentication_service.py uses:
#
# predicted_user == username
#
# Therefore:
#
# predicted_user == actual_user
#       -> ACCEPT
#
# predicted_user != actual_user
#       -> REJECT
#
# ==========================================================

genuine_attempts = len(
    actual_users
)

genuine_accepts = 0

genuine_rejects = 0


for i in range(
    genuine_attempts
):

    if (
        fusion_predictions[i]
        ==
        actual_users[i]
    ):

        genuine_accepts += 1

    else:

        genuine_rejects += 1


# ==========================================================
# IMPOSTOR AUTHENTICATION
# ==========================================================
#
# For each test sample:
#
# Actual user = A
#
# We test claims:
#
# B, C, D, ... all other users.
#
# If the fusion prediction equals
# the claimed impostor identity:
#
#     FALSE ACCEPT
#
# Otherwise:
#
#     TRUE REJECT
#
# ==========================================================

impostor_attempts = 0

impostor_accepts = 0

impostor_rejects = 0


for i in range(
    len(actual_users)
):

    actual_user = actual_users[i]

    predicted_user = fusion_predictions[i]

    for claimed_user in usernames:

        claimed_user = str(
            claimed_user
        )

        # Skip the genuine identity.

        if claimed_user == actual_user:

            continue

        impostor_attempts += 1

        # --------------------------------------------------
        # Authentication decision
        # --------------------------------------------------

        if predicted_user == claimed_user:

            # Impostor was accepted.

            impostor_accepts += 1

        else:

            # Impostor was rejected.

            impostor_rejects += 1


# ==========================================================
# FAR
# ==========================================================

if impostor_attempts > 0:

    far = (
        impostor_accepts
        /
        impostor_attempts
    )

else:

    far = 0.0


# ==========================================================
# FRR
# ==========================================================

if genuine_attempts > 0:

    frr = (
        genuine_rejects
        /
        genuine_attempts
    )

else:

    frr = 0.0


# ==========================================================
# TAR
# ==========================================================

if genuine_attempts > 0:

    tar = (
        genuine_accepts
        /
        genuine_attempts
    )

else:

    tar = 0.0


# ==========================================================
# TRR
# ==========================================================

if impostor_attempts > 0:

    trr = (
        impostor_rejects
        /
        impostor_attempts
    )

else:

    trr = 0.0


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()
print("=" * 70)
print("MODEL IDENTIFICATION RESULTS")
print("=" * 70)

print()

print(
    f"LSTM Accuracy   : "
    f"{lstm_accuracy * 100:.2f}%"
)

print(
    f"GRU Accuracy    : "
    f"{gru_accuracy * 100:.2f}%"
)

print(
    f"Fusion Accuracy : "
    f"{fusion_accuracy * 100:.2f}%"
)

print()


print("=" * 70)
print("AUTHENTICATION RESULTS")
print("=" * 70)

print()

print(
    f"Genuine Attempts : "
    f"{genuine_attempts}"
)

print(
    f"Genuine Accepts  : "
    f"{genuine_accepts}"
)

print(
    f"Genuine Rejects  : "
    f"{genuine_rejects}"
)

print()

print(
    f"Impostor Attempts : "
    f"{impostor_attempts}"
)

print(
    f"Impostor Accepts  : "
    f"{impostor_accepts}"
)

print(
    f"Impostor Rejects  : "
    f"{impostor_rejects}"
)

print()


print("-" * 70)

print(
    f"FAR "
    f"(False Acceptance Rate) : "
    f"{far * 100:.4f}%"
)

print(
    f"FRR "
    f"(False Rejection Rate)  : "
    f"{frr * 100:.4f}%"
)

print(
    f"TAR "
    f"(True Acceptance Rate)  : "
    f"{tar * 100:.4f}%"
)

print(
    f"TRR "
    f"(True Rejection Rate)   : "
    f"{trr * 100:.4f}%"
)

print()

print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)