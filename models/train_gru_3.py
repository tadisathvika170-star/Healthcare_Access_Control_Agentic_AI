"""
train_gru_3.py

Train Feature-Enhanced GRU using
3-repetition sequences.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from models.gru_model import FeatureEnhancedGRU


# ==========================================================
# CONFIG
# ==========================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 30

SEQUENCE_LENGTH = 3
INPUT_SIZE = 31
NUM_CLASSES = 51

MODEL_PATH = "models/gru_model.pth"


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("3-REPETITION FEATURE-ENHANCED GRU TRAINING")
print("=" * 70)

print(
    f"Device: {device}"
)


# ==========================================================
# LOAD DATA
# ==========================================================

X_train = np.load(
    "dataset/X_train.npy"
)

y_train = np.load(
    "dataset/y_train.npy"
)

X_test = np.load(
    "dataset/X_test.npy"
)

y_test = np.load(
    "dataset/y_test.npy"
)


print(
    f"X_train: {X_train.shape}"
)

print(
    f"X_test : {X_test.shape}"
)


# ==========================================================
# VALIDATE
# ==========================================================

assert X_train.shape[1] == 3
assert X_train.shape[2] == 31

assert X_test.shape[1] == 3
assert X_test.shape[2] == 31


# ==========================================================
# TENSORS
# ==========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ==========================================================
# LOADERS
# ==========================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================================
# MODEL
# ==========================================================

model = FeatureEnhancedGRU(
    input_size=31,
    hidden_size=128,
    num_layers=1,
    num_classes=NUM_CLASSES,
    dropout=0.0,
).to(device)


# ==========================================================
# LOSS
# ==========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==========================================================
# SCHEDULER
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ==========================================================
# TRAIN
# ==========================================================

best_accuracy = 0.0

best_model = copy.deepcopy(
    model.state_dict()
)

start_time = time.time()


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_labels = []


    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            inputs
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_labels.extend(
            labels.cpu().numpy()
        )


    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )


    # ======================================================
    # TEST / VALIDATION
    # ======================================================

    model.eval()

    test_predictions = []
    test_labels = []

    test_loss = 0.0


    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels
            )

            test_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )

            test_predictions.extend(
                predictions.cpu().numpy()
            )

            test_labels.extend(
                labels.cpu().numpy()
            )


    test_accuracy = accuracy_score(
        test_labels,
        test_predictions
    )


    scheduler.step(
        test_accuracy
    )


    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        best_model = copy.deepcopy(
            model.state_dict()
        )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {running_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )


# ==========================================================
# SAVE BEST MODEL
# ==========================================================

model.load_state_dict(
    best_model
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


training_time = (
    time.time() -
    start_time
)


print("\n" + "=" * 70)
print("GRU TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best test accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)

print("=" * 70)