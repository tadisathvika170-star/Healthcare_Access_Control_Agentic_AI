"""
train_lstm.py

Train the LSTM model for
Behavioral Biometrics Authentication.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score

from config import (
    LSTM_MODEL_PATH,
)

from dataset.dataset import KeystrokeDataset
from models.lstm_model import LSTMModel


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("LSTM TRAINING")
print("=" * 60)
print(f"Device : {device}")
print()


# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20


# ==========================================================
# Dataset
# ==========================================================

dataset = KeystrokeDataset(batch_size=BATCH_SIZE)

train_loader = dataset.train_loader()

test_loader = dataset.test_loader()

print(f"Training Batches : {len(train_loader)}")
print(f"Testing Batches  : {len(test_loader)}")
print()


# ==========================================================
# Number of Classes
# ==========================================================

num_classes = len(torch.unique(dataset.y_train))

print(f"Detected Classes : {num_classes}")
print()


# ==========================================================
# Model
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
# Loss & Optimizer
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==========================================================
# Variables
# ==========================================================

best_accuracy = 0.0

best_model = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_accuracy": [],
}

start_time = time.time()


# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    train_predictions = []

    train_labels = []

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_predictions.extend(
            predicted.cpu().numpy()
        )

        train_labels.extend(
            labels.cpu().numpy()
        )

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions,
    )

    history["train_loss"].append(
        running_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    test_predictions = []

    test_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            test_predictions.extend(
                predicted.cpu().numpy()
            )

            test_labels.extend(
                labels.cpu().numpy()
            )

    test_accuracy = accuracy_score(
        test_labels,
        test_predictions,
    )

    history["test_accuracy"].append(
        test_accuracy
    )
        # ==========================================================
    # Save Best Model
    # ==========================================================

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        best_model = copy.deepcopy(model.state_dict())

        torch.save(
            best_model,
            LSTM_MODEL_PATH,
        )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Train Loss: {running_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Acc: {test_accuracy * 100:.2f}%"
    )


# ==========================================================
# Training Complete
# ==========================================================

training_time = time.time() - start_time

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best Test Accuracy : {best_accuracy * 100:.2f}%")
print(f"Training Time      : {training_time:.2f} seconds")
print(f"Model Saved To     : {Path(LSTM_MODEL_PATH).resolve()}")

print("=" * 60)

# ==========================================================
# Load Best Model
# ==========================================================

model.load_state_dict(best_model)

print("\nBest LSTM model loaded successfully.")

# ==========================================================
# Training History
# ==========================================================

print("\nTraining Summary")

print("-" * 60)

for i in range(EPOCHS):

    print(
        f"Epoch {i + 1:02d} | "
        f"Loss: {history['train_loss'][i]:.4f} | "
        f"Train Acc: {history['train_accuracy'][i] * 100:.2f}% | "
        f"Test Acc: {history['test_accuracy'][i] * 100:.2f}%"
    )

print("-" * 60)

print("\nLSTM Training Finished Successfully.")