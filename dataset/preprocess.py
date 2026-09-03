import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_PATH,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
    SCALER_PATH,
)


class DataPreprocessor:
    """
    Loads and preprocesses the CMU Keystroke Dynamics dataset.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_dataset(self):
        """
        Load the dataset.
        """
        print("Loading dataset...")

        df = pd.read_csv(DATASET_PATH)

        print(f"Dataset Shape : {df.shape}")

        return df

    def preprocess(self):

        df = self.load_dataset()

        # -----------------------------
        # Remove Missing Values
        # -----------------------------
        df = df.dropna()

        print(f"Shape after removing missing values : {df.shape}")

        # -----------------------------
        # Encode Users
        # -----------------------------
        df["subject"] = self.label_encoder.fit_transform(df["subject"])

        # -----------------------------
        # Features and Labels
        # -----------------------------
        X = df.drop(columns=["subject"])

        y = df["subject"]

        # -----------------------------
        # Scale Features
        # -----------------------------
        X_scaled = self.scaler.fit_transform(X)

        # -----------------------------
        # Save Scaler
        # -----------------------------
        joblib.dump(self.scaler, SCALER_PATH)

        print("Scaler saved.")

        # -----------------------------
        # Save Label Encoder
        # -----------------------------
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)

        print("Label Encoder saved.")

        # -----------------------------
        # Train Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print()

        print("Training Samples :", len(X_train))
        print("Testing Samples  :", len(X_test))

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )


if __name__ == "__main__":

    processor = DataPreprocessor()

    processor.preprocess()