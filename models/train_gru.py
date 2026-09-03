print("TRAIN_GRU FILE IS RUNNING")
"""
train_gru.py

Train the GRU model for keystroke authentication.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import GRU_MODEL_PATH
from dataset.dataset import KeystrokeDataset
from models.gru_model import GRUModel


# ---------------------------------------
# Device Configuration
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}\n")


# ---------------------------------------
# Hyperparameters
# ---------------------------------------

EPOCHS = 20
LEARNING_RATE = 0.001


# ---------------------------------------
# Dataset
# ---------------------------------------

dataset = KeystrokeDataset(batch_size=32)

train_loader = dataset.train_loader()

test_loader = dataset.test_loader()


# ---------------------------------------
# Model
# ---------------------------------------

model = GRUModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------
# Training
# ---------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# ---------------------------------------
# Save Model
# ---------------------------------------

torch.save(
    model.state_dict(),
    GRU_MODEL_PATH
)

print("\nGRU Model Saved Successfully.")
