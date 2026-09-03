"""
test_fusion.py

Evaluate the LSTM + GRU Fusion model
using the existing trained models.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    DATASET_PATH,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
)

from predict.predictor import Predictor


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("LSTM + GRU FUSION MODEL EVALUATION")
print("=" * 70)
print()


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

print("Loading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print(
    f"Number of Users : "
    f"{len(label_encoder.classes_)}"
)

print()


# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset Shape : {df.shape}"
)

# Remove missing values

df = df.dropna().reset_index(
    drop=True
)

print(
    f"After Removing Missing Values : "
    f"{df.shape}"
)

print()


# ==========================================================
# FIND USER COLUMN
# ==========================================================

if "subject" in df.columns:

    target_column = "subject"

elif "user" in df.columns:

    target_column = "user"

elif "username" in df.columns:

    target_column = "username"

else:

    raise ValueError(
        "User/subject column not found."
    )


# ==========================================================
# FEATURES AND LABELS
# ==========================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


print(
    f"Number of Features : {X.shape[1]}"
)

print()


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

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
# PREDICTION STORAGE
# ==========================================================

actual_users = []

lstm_predictions = []

gru_predictions = []

fusion_predictions = []

lstm_confidences = []

gru_confidences = []

fusion_confidences = []

agreements = []


# ==========================================================
# EVALUATION
# ==========================================================

print("Evaluating Fusion Model...")
print()

for index in range(
    len(X_test)
):

    # ------------------------------------------------------
    # RAW FEATURES
    # ------------------------------------------------------

    features = X_test.iloc[index].to_numpy(
        dtype=np.float64
    )

    actual_user = str(
        y_test.iloc[index]
    )

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    result = predictor.predict(
        features
    )

    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    actual_users.append(
        actual_user
    )

    lstm_predictions.append(
        result["lstm_prediction"]
    )

    gru_predictions.append(
        result["gru_prediction"]
    )

    fusion_predictions.append(
        result["predicted_user"]
    )

    lstm_confidences.append(
        result["lstm_confidence"]
    )

    gru_confidences.append(
        result["gru_confidence"]
    )

    fusion_confidences.append(
        result["confidence"]
    )

    agreements.append(
        result["agreement"]
    )


# ==========================================================
# ACCURACY
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
# MODEL AGREEMENT
# ==========================================================

agreement_percentage = (
    sum(agreements)
    /
    len(agreements)
) * 100


# ==========================================================
# AVERAGE CONFIDENCE
# ==========================================================

average_lstm_confidence = (
    np.mean(lstm_confidences)
    * 100
)

average_gru_confidence = (
    np.mean(gru_confidences)
    * 100
)

average_fusion_confidence = (
    np.mean(fusion_confidences)
    * 100
)


# ==========================================================
# CORRECT PREDICTIONS
# ==========================================================

lstm_correct = sum(
    1
    for actual, predicted
    in zip(
        actual_users,
        lstm_predictions
    )
    if actual == predicted
)

gru_correct = sum(
    1
    for actual, predicted
    in zip(
        actual_users,
        gru_predictions
    )
    if actual == predicted
)

fusion_correct = sum(
    1
    for actual, predicted
    in zip(
        actual_users,
        fusion_predictions
    )
    if actual == predicted
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()
print("=" * 70)
print("FUSION MODEL RESULTS")
print("=" * 70)

print()

print(
    f"Total Test Samples       : "
    f"{len(actual_users)}"
)

print()

print(
    f"LSTM Correct Predictions   : "
    f"{lstm_correct}"
)

print(
    f"GRU Correct Predictions    : "
    f"{gru_correct}"
)

print(
    f"Fusion Correct Predictions : "
    f"{fusion_correct}"
)

print()

print("-" * 70)

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

print("-" * 70)

print(
    f"Average LSTM Confidence   : "
    f"{average_lstm_confidence:.2f}%"
)

print(
    f"Average GRU Confidence    : "
    f"{average_gru_confidence:.2f}%"
)

print(
    f"Average Fusion Confidence : "
    f"{average_fusion_confidence:.2f}%"
)

print()

print(
    f"LSTM-GRU Agreement : "
    f"{agreement_percentage:.2f}%"
)

print()

print("=" * 70)
print("FUSION EVALUATION COMPLETED")
print("=" * 70)