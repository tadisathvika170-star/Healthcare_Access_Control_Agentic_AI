"""
test_lstm.py

Evaluate the already-trained LSTM model.

This file DOES NOT train the model and DOES NOT overwrite
lstm_model.pth.

It calculates:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
"""

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset.dataset import KeystrokeDataset
from models.lstm_model import LSTMModel
from config import LSTM_MODEL_PATH


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("LSTM MODEL EVALUATION")
print("=" * 60)
print(f"Device : {device}")
print()


# ==========================================================
# Dataset
# ==========================================================

print("Loading dataset...")

dataset = KeystrokeDataset(
    batch_size=32
)

test_loader = dataset.test_loader()

print(
    f"Testing Batches : {len(test_loader)}"
)

print(
    f"Test Samples    : {len(dataset.y_test)}"
)

print()


# ==========================================================
# Number of Classes
# ==========================================================

num_classes = len(
    torch.unique(dataset.y_train)
)

print(
    f"Number of Classes : {num_classes}"
)

print()


# ==========================================================
# Create LSTM Model
# ==========================================================

model = LSTMModel(
    input_size=33,
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.2,
)

model = model.to(device)


# ==========================================================
# Load Existing Trained Model
# ==========================================================

print("Loading trained LSTM model...")

state_dict = torch.load(
    LSTM_MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print(
    f"Model Loaded From : {LSTM_MODEL_PATH}"
)

print()


# ==========================================================
# Evaluation
# ==========================================================

all_predictions = []
all_labels = []

print("Evaluating model...")
print()

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(
            inputs
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================================
# Convert to NumPy
# ==========================================================

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ==========================================================
# Display Results
# ==========================================================

print("=" * 60)
print("LSTM EVALUATION RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print("=" * 60)


# ==========================================================
# Correct / Incorrect Predictions
# ==========================================================

correct = np.sum(
    all_labels == all_predictions
)

incorrect = np.sum(
    all_labels != all_predictions
)

total = len(
    all_labels
)

print()
print("Prediction Summary")
print("-" * 60)

print(
    f"Total Test Samples       : {total}"
)

print(
    f"Correct Predictions      : {correct}"
)

print(
    f"Incorrect Predictions    : {incorrect}"
)

print()


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix Shape:")
print(cm.shape)

print()

print(
    "LSTM evaluation completed successfully."
)

print("=" * 60)