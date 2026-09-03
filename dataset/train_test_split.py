"""
train_test_split.py

Splits the keystroke dataset into training and testing sets.
"""

from sklearn.model_selection import train_test_split

from dataset.preprocess import preprocess_dataset
from config import RANDOM_STATE


def get_train_test_data():
    """
    Returns:
        X_train
        X_test
        y_train
        y_test
    """

    X, y, scaler, label_encoder = preprocess_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True
    )

    print("\nTrain/Test Split Completed")

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_train_test_data()

    print("\nShapes")

    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)
    print("y_train :", y_train.shape)
    print("y_test  :", y_test.shape)