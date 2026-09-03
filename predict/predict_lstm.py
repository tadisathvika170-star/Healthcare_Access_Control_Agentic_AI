"""
predict_lstm.py

Loads the trained LSTM model and predicts
the user based on keystroke features.

The scaler receives a pandas DataFrame with
the original feature names to avoid the
StandardScaler feature-name warning.
"""

import joblib
import numpy as np
import pandas as pd
import torch

from config import (
    LSTM_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
)

from models.lstm_model import LSTMModel


class LSTMPredictor:

    def __init__(self):

        # ==================================================
        # Device
        # ==================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==================================================
        # Load Scaler
        # ==================================================

        self.scaler = joblib.load(
            SCALER_PATH
        )

        # ==================================================
        # Load Label Encoder
        # ==================================================

        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        # ==================================================
        # Number of Classes
        # ==================================================

        self.num_classes = len(
            self.label_encoder.classes_
        )

        # ==================================================
        # Get Original Feature Names
        # ==================================================

        if hasattr(
            self.scaler,
            "feature_names_in_"
        ):

            self.feature_names = list(
                self.scaler.feature_names_in_
            )

        else:

            self.feature_names = None

        # ==================================================
        # Load LSTM Model
        # ==================================================

        self.model = LSTMModel(
            input_size=33,
            hidden_size=64,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.2,
        )

        # ==================================================
        # Load Trained Model
        # ==================================================

        self.model.load_state_dict(
            torch.load(
                LSTM_MODEL_PATH,
                map_location=self.device
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        features
    ):
        """
        Predict user from 33 keystroke features.

        Parameters
        ----------
        features:
            list, NumPy array, pandas Series,
            or one-row pandas DataFrame.

        Returns
        -------
        tuple:
            predicted_user, confidence
        """

        # ==================================================
        # Handle DataFrame
        # ==================================================

        if isinstance(
            features,
            pd.DataFrame
        ):

            if len(features) != 1:

                raise ValueError(
                    "LSTM predictor accepts "
                    "one sample at a time."
                )

            # If DataFrame already contains
            # feature names, preserve them.

            if self.feature_names is not None:

                missing_features = [
                    feature
                    for feature in self.feature_names
                    if feature not in features.columns
                ]

                if missing_features:

                    raise ValueError(
                        "Missing feature(s): "
                        + str(missing_features)
                    )

                features = features[
                    self.feature_names
                ]

                features_array = (
                    features.iloc[0]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

            else:

                features_array = (
                    features.iloc[0]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

        # ==================================================
        # Handle pandas Series
        # ==================================================

        elif isinstance(
            features,
            pd.Series
        ):

            features_array = (
                features.to_numpy(
                    dtype=np.float64
                )
            )

        # ==================================================
        # Handle list / NumPy array
        # ==================================================

        else:

            features_array = np.asarray(
                features,
                dtype=np.float64
            )

        # ==================================================
        # Flatten
        # ==================================================

        features_array = (
            features_array.reshape(-1)
        )

        # ==================================================
        # Feature Count Check
        # ==================================================

        if len(features_array) != 33:

            raise ValueError(
                "LSTM expects exactly 33 "
                f"features, but received "
                f"{len(features_array)}."
            )

        # ==================================================
        # Create DataFrame for Scaler
        # ==================================================
        #
        # This is the important part that removes:
        #
        # UserWarning:
        # X does not have valid feature names
        #
        # ==================================================

        if self.feature_names is not None:

            features_df = pd.DataFrame(
                [features_array],
                columns=self.feature_names
            )

        else:

            features_df = pd.DataFrame(
                [features_array]
            )

        # ==================================================
        # Scale
        # ==================================================

        scaled_features = (
            self.scaler.transform(
                features_df
            )
        )

        # ==================================================
        # Convert to Tensor
        # ==================================================
        #
        # Shape:
        #
        # (batch, sequence, features)
        #
        # (1, 1, 33)
        #
        # ==================================================

        tensor = torch.FloatTensor(
            scaled_features
        ).unsqueeze(1)

        tensor = tensor.to(
            self.device
        )

        # ==================================================
        # Model Prediction
        # ==================================================

        with torch.no_grad():

            outputs = self.model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            # ==================================================
            # Convert Prediction to Username
            # ==================================================

            predicted_user = (
                self.label_encoder
                .inverse_transform(
                    prediction
                    .cpu()
                    .numpy()
                )[0]
            )

            # ==================================================
            # Confidence
            # ==================================================

            confidence = float(
                confidence
                .cpu()
                .item()
            )

        return (
            predicted_user,
            confidence
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LSTM PREDICTOR TEST")
    print("=" * 60)

    try:

        predictor = LSTMPredictor()

        print()
        print(
            "LSTM model loaded successfully."
        )

        print(
            f"Number of Classes : "
            f"{predictor.num_classes}"
        )

        if predictor.feature_names is not None:

            print(
                f"Number of Features : "
                f"{len(predictor.feature_names)}"
            )

        else:

            print(
                "Number of Features : 33"
            )

        # --------------------------------------------------
        # Test sample
        # --------------------------------------------------

        sample = np.random.rand(33)

        user, confidence = (
            predictor.predict(sample)
        )

        print()
        print(
            f"Predicted User : {user}"
        )

        print(
            f"Confidence     : "
            f"{confidence * 100:.2f}%"
        )

        print()
        print(
            "LSTM predictor test completed successfully."
        )

    except Exception as e:

        print()
        print(
            "LSTM predictor test failed."
        )

        print(
            f"Error : {e}"
        )

    print()
    print("=" * 60)