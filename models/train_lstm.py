"""Train the LSTM model for single-repetition keystroke authentication."""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import LSTMModel

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
SEQUENCE_LENGTH = 1
INPUT_SIZE = 31
NUM_CLASSES = 51
MODEL_PATH = "models/lstm_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("SINGLE-REPETITION LSTM TRAINING")
print(f"Device: {device}")

X_train = np.load("dataset/X_train.npy")
y_train = np.load("dataset/y_train.npy")
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")

assert X_train.shape[1:] == (SEQUENCE_LENGTH, INPUT_SIZE)
assert X_test.shape[1:] == (SEQUENCE_LENGTH, INPUT_SIZE)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False,
)

model = LSTMModel(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES, dropout=0.2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_accuracy = 0.0
best_model = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_predictions, train_labels = [], []
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        train_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    model.eval()
    test_predictions, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}% | Test Acc: {test_accuracy * 100:.2f}%")

model.load_state_dict(best_model)
torch.save(model.state_dict(), MODEL_PATH)
print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETED")
print(f"Best test accuracy: {best_accuracy * 100:.2f}%")
print(f"Training time: {time.time() - start_time:.2f} seconds")
print(f"Model saved to: {MODEL_PATH}")
print("=" * 70)
