import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from dataset.dataset import load_data
from models.gru_model import FeatureEnhancedGRU


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

PATIENCE = 6
GRADIENT_CLIP = 1.0

NUM_CLASSES = 51
INPUT_SIZE = 31
HIDDEN_SIZE = 128
NUM_LAYERS = 1

MODEL_PATH = "models/gru_model.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Load dataset
# ============================================================

X_train, y_train, X_test, y_test = load_data()

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# Convert NumPy → PyTorch tensors
# ============================================================

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


# ============================================================
# Train / Validation split
# ============================================================

full_train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples:       {len(test_dataset)}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

model = FeatureEnhancedGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=0.0
).to(device)

print("\nModel:")
print(model)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            predictions == labels
        ).sum().item()

        total_train += labels.size(0)

    train_loss = total_train_loss / total_train
    train_accuracy = (
        correct_train / total_train
    ) * 100


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            total_val_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_val += (
                predictions == labels
            ).sum().item()

            total_val += labels.size(0)

    val_loss = total_val_loss / total_val
    val_accuracy = (
        correct_val / total_val
    ) * 100


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  -> Best model saved "
            f"(Validation Accuracy: "
            f"{val_accuracy:.2f}%)"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break


# ============================================================
# Training time
# ============================================================

training_time = time.time() - start_time

print(
    f"\nTraining time: {training_time:.2f} seconds"
)


# ============================================================
# Load best model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Final test evaluation
# ============================================================

total_test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        total_test_loss += (
            loss.item() * inputs.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_test += (
            predictions == labels
        ).sum().item()

        total_test += labels.size(0)

test_loss = total_test_loss / total_test

test_accuracy = (
    correct_test / total_test
) * 100


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("FEATURE-ENHANCED GRU RESULTS")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    "Previous tuned GRU: 87.06%"
)

print(
    f"Improvement: "
    f"{test_accuracy - 87.06:+.2f} percentage points"
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

print("=" * 60)