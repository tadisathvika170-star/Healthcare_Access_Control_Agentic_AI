import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from dataset.dataset import load_data
from models.lstm_model import LSTMModel


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
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Load data
# ============================================================

X_train, y_train, X_test, y_test = load_data()

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# Convert to tensors
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
# Fixed train / validation split
# ============================================================

full_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
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
# Hyperparameter configurations
# ============================================================

configs = [
    {
        "name": "LSTM_64_L1",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 1e-4
    },
    {
        "name": "LSTM_128_L1",
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 1e-4
    },
    {
        "name": "LSTM_256_L1",
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 1e-4
    },
    {
        "name": "LSTM_128_L2",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 1e-4
    },
    {
        "name": "LSTM_128_L1_LR0005",
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4
    },
    {
        "name": "LSTM_128_L1_WD001",
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.001
    }
]


# ============================================================
# General training settings
# ============================================================

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 6
GRADIENT_CLIP = 1.0

INPUT_SIZE = 31
NUM_CLASSES = 51


# ============================================================
# Results storage
# ============================================================

results = []

overall_best_accuracy = 0.0
overall_best_config = None
overall_best_state = None


# ============================================================
# Hyperparameter tuning
# ============================================================

for config_number, config in enumerate(configs, start=1):

    print("\n" + "=" * 70)
    print(
        f"CONFIGURATION {config_number}/{len(configs)}: "
        f"{config['name']}"
    )
    print("=" * 70)

    print(
        f"Hidden Size: {config['hidden_size']}"
    )

    print(
        f"Layers: {config['num_layers']}"
    )

    print(
        f"Dropout: {config['dropout']}"
    )

    print(
        f"Learning Rate: {config['learning_rate']}"
    )

    print(
        f"Weight Decay: {config['weight_decay']}"
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=config["dropout"]
    ).to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )


    # --------------------------------------------------------
    # Best configuration values
    # --------------------------------------------------------

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        correct_train = 0
        total_train = 0
        total_train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

            total_train_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_train += (
                predictions == labels
            ).sum().item()

            total_train += labels.size(0)


        train_accuracy = (
            correct_train / total_train
        ) * 100


        train_loss = (
            total_train_loss / total_train
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        correct_val = 0
        total_val = 0
        total_val_loss = 0.0

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


        val_accuracy = (
            correct_val / total_val
        ) * 100


        val_loss = (
            total_val_loss / total_val
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]


        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Acc: {val_accuracy:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {current_lr:.6f}"
        )


        # ----------------------------------------------------
        # Best validation score
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break


    training_time = time.time() - start_time


    # --------------------------------------------------------
    # Store configuration result
    # --------------------------------------------------------

    results.append(
        {
            "name": config["name"],
            "hidden_size": config["hidden_size"],
            "layers": config["num_layers"],
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "training_time": training_time
        }
    )


    print(
        f"\n{config['name']} finished"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # Track overall best
    # --------------------------------------------------------

    if best_val_accuracy > overall_best_accuracy:

        overall_best_accuracy = best_val_accuracy

        overall_best_config = config.copy()

        overall_best_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }


# ============================================================
# Print tuning results
# ============================================================

print("\n" + "=" * 80)
print("LSTM HYPERPARAMETER TUNING RESULTS")
print("=" * 80)

for result in results:

    print(
        f"{result['name']:25s} "
        f"Val Accuracy: "
        f"{result['best_val_accuracy']:.2f}% "
        f"| Epoch: {result['best_epoch']:2d} "
        f"| Time: "
        f"{result['training_time']:.2f}s"
    )


# ============================================================
# Best configuration
# ============================================================

print("\n" + "=" * 80)
print("BEST LSTM CONFIGURATION")
print("=" * 80)

print(
    f"Configuration: "
    f"{overall_best_config['name']}"
)

print(
    f"Hidden Size: "
    f"{overall_best_config['hidden_size']}"
)

print(
    f"Layers: "
    f"{overall_best_config['num_layers']}"
)

print(
    f"Dropout: "
    f"{overall_best_config['dropout']}"
)

print(
    f"Learning Rate: "
    f"{overall_best_config['learning_rate']}"
)

print(
    f"Weight Decay: "
    f"{overall_best_config['weight_decay']}"
)

print(
    f"Best Validation Accuracy: "
    f"{overall_best_accuracy:.2f}%"
)


# ============================================================
# Save best tuned model
# ============================================================

best_model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=overall_best_config["hidden_size"],
    num_layers=overall_best_config["num_layers"],
    num_classes=NUM_CLASSES,
    dropout=overall_best_config["dropout"]
).to(device)

best_model.load_state_dict(
    overall_best_state
)

best_model.eval()


torch.save(
    best_model.state_dict(),
    "models/lstm_tuned.pth"
)

print(
    "\nBest tuned model saved to:"
    " models/lstm_tuned.pth"
)


# ============================================================
# Final test evaluation
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

correct_test = 0
total_test = 0
total_test_loss = 0.0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = best_model(inputs)

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


test_accuracy = (
    correct_test / total_test
) * 100

test_loss = (
    total_test_loss / total_test
)


# ============================================================
# Final results
# ============================================================

print("\n" + "=" * 80)
print("FINAL TUNED LSTM RESULTS")
print("=" * 80)

print(
    f"Best Validation Accuracy: "
    f"{overall_best_accuracy:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    f"Current Feature-Enhanced GRU: 92.16%"
)

print(
    f"Original LSTM: 92.35%"
)

print(
    f"Tuned LSTM vs Feature-Enhanced GRU: "
    f"{test_accuracy - 92.16:+.2f} percentage points"
)

print(
    f"Tuned LSTM vs Original LSTM: "
    f"{test_accuracy - 92.35:+.2f} percentage points"
)

print("=" * 80)