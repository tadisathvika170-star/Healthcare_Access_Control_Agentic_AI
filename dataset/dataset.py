import numpy as np
import os


DATA_DIR = "dataset"


def load_data():

    X_train_path = os.path.join(
        DATA_DIR,
        "X_train.npy"
    )

    y_train_path = os.path.join(
        DATA_DIR,
        "y_train.npy"
    )

    X_test_path = os.path.join(
        DATA_DIR,
        "X_test.npy"
    )

    y_test_path = os.path.join(
        DATA_DIR,
        "y_test.npy"
    )

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data()

    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    print("=" * 60)