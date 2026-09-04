"""
FINAL LSTM + FEATURE-ENHANCED GRU FUSION

Pipeline:
10 repetitions
    ↓
31 behavioral features
    ↓
StandardScaler
    ↓
LSTM + Feature-Enhanced GRU
    ↓
Equal-weight probability fusion
    ↓
Identity + confidence + model agreement
"""

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

LSTM_PATH = MODEL_DIR / "lstm_model.pth"
GRU_PATH = MODEL_DIR / "gru_model.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ==========================================================
# FINAL CONFIGURATION
# ==========================================================

SEQUENCE_LENGTH = 10
INPUT_SIZE = 31
NUM_CLASSES = 51


# ==========================================================
# LSTM MODEL
# EXACT MATCH TO TRAINED CHECKPOINT
# ==========================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=31,
        hidden_size=64,
        num_layers=2,
        num_classes=51,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.classifier(
            last_output
        )


# ==========================================================
# FEATURE-ENHANCED GRU
# EXACT MATCH TO TRAINED CHECKPOINT
# ==========================================================

class FeatureEnhancedGRU(nn.Module):

    def __init__(
        self,
        input_size=31,
        hidden_size=128,
        num_layers=1,
        num_classes=51,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # IMPORTANT:
        # The trained checkpoint uses the name
        # "statistics", not "statistical_branch".

        self.statistics = nn.Sequential(
            nn.Linear(
                input_size * 5,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
        )

        # 128 GRU features + 64 statistical features = 192

        self.classifier = nn.Sequential(
            nn.Linear(
                192,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        gru_output, _ = self.gru(x)

        gru_features = gru_output[:, -1, :]

        # --------------------------------------------------
        # Statistical features
        # --------------------------------------------------

        mean_features = torch.mean(
            x,
            dim=1,
        )

        std_features = torch.std(
            x,
            dim=1,
            unbiased=False,
        )

        min_features = torch.min(
            x,
            dim=1,
        ).values

        max_features = torch.max(
            x,
            dim=1,
        ).values

        range_features = (
            max_features
            - min_features
        )

        statistical_features = torch.cat(
            [
                mean_features,
                std_features,
                min_features,
                max_features,
                range_features,
            ],
            dim=1,
        )

        statistical_features = self.statistics(
            statistical_features
        )

        # --------------------------------------------------
        # Fusion inside GRU
        # --------------------------------------------------

        fused_features = torch.cat(
            [
                gru_features,
                statistical_features,
            ],
            dim=1,
        )

        return self.classifier(
            fused_features
        )


# ==========================================================
# FUSION PREDICTOR
# ==========================================================

class FusionPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Verify required files
        # --------------------------------------------------

        for path in (
            LSTM_PATH,
            GRU_PATH,
            SCALER_PATH,
        ):

            if not path.exists():

                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        # --------------------------------------------------
        # Load LSTM
        # --------------------------------------------------

        self.lstm_model = LSTMModel()

        self.lstm_model.load_state_dict(
            torch.load(
                LSTM_PATH,
                map_location=self.device,
            )
        )

        self.lstm_model.to(
            self.device
        )

        self.lstm_model.eval()

        # --------------------------------------------------
        # Load GRU
        # --------------------------------------------------

        self.gru_model = FeatureEnhancedGRU()

        self.gru_model.load_state_dict(
            torch.load(
                GRU_PATH,
                map_location=self.device,
            )
        )

        self.gru_model.to(
            self.device
        )

        self.gru_model.eval()

        # --------------------------------------------------
        # Load scaler
        # --------------------------------------------------

        self.scaler = joblib.load(
            SCALER_PATH
        )

    # ======================================================
    # PREPARE SEQUENCE
    # ======================================================

    def _prepare_sequence(
        self,
        sequence,
    ):

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_shape = (
            SEQUENCE_LENGTH,
            INPUT_SIZE,
        )

        if sequence.ndim != 2:

            raise ValueError(
                "Expected a 2D keystroke sequence. "
                f"Expected shape {expected_shape}; "
                f"received {sequence.shape}."
            )

        if sequence.shape != expected_shape:

            raise ValueError(
                "Invalid keystroke sequence shape. "
                f"Expected {expected_shape}; "
                f"received {sequence.shape}."
            )

        scaled_sequence = self.scaler.transform(
            sequence
        )

        tensor = torch.tensor(
            scaled_sequence,
            dtype=torch.float32,
        )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.to(
            self.device
        )

    # ======================================================
    # PREDICTION
    # ======================================================

    @torch.no_grad()
    def predict(
        self,
        sequence,
    ):

        x = self._prepare_sequence(
            sequence
        )

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        lstm_logits = self.lstm_model(
            x
        )

        lstm_probabilities = torch.softmax(
            lstm_logits,
            dim=1,
        )

        lstm_confidence, lstm_prediction = torch.max(
            lstm_probabilities,
            dim=1,
        )

        # --------------------------------------------------
        # GRU
        # --------------------------------------------------

        gru_logits = self.gru_model(
            x
        )

        gru_probabilities = torch.softmax(
            gru_logits,
            dim=1,
        )

        gru_confidence, gru_prediction = torch.max(
            gru_probabilities,
            dim=1,
        )

        # --------------------------------------------------
        # EQUAL-WEIGHT FUSION
        # --------------------------------------------------

        fusion_probabilities = (
            lstm_probabilities
            + gru_probabilities
        ) / 2.0

        fusion_confidence, fusion_prediction = torch.max(
            fusion_probabilities,
            dim=1,
        )

        # --------------------------------------------------
        # MODEL AGREEMENT
        # --------------------------------------------------

        agreement = (
            lstm_prediction.item()
            == gru_prediction.item()
        )

        return {
            "predicted_user":
                int(
                    fusion_prediction.item()
                ),

            "confidence":
                float(
                    fusion_confidence.item()
                ),

            "lstm_prediction":
                int(
                    lstm_prediction.item()
                ),

            "lstm_confidence":
                float(
                    lstm_confidence.item()
                ),

            "gru_prediction":
                int(
                    gru_prediction.item()
                ),

            "gru_confidence":
                float(
                    gru_confidence.item()
                ),

            "agreement":
                bool(
                    agreement
                ),
        }


# ==========================================================
# BACKWARD COMPATIBILITY
# ==========================================================

def predict_fusion(
    sequence,
):

    predictor = FusionPredictor()

    return predictor.predict(
        sequence
    )


# ==========================================================
# FINAL COMPATIBILITY TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FINAL FUSION PIPELINE TEST")
    print("=" * 60)

    print(
        "Sequence length :",
        SEQUENCE_LENGTH,
    )

    print(
        "Input features  :",
        INPUT_SIZE,
    )

    print(
        "Number of users :",
        NUM_CLASSES,
    )

    print()

    predictor = FusionPredictor()

    sample = np.random.randn(
        SEQUENCE_LENGTH,
        INPUT_SIZE,
    ).astype(
        np.float32
    )

    result = predictor.predict(
        sample
    )

    print(
        "Test sequence shape:",
        sample.shape,
    )

    print()

    print(
        "Predicted User   :",
        result["predicted_user"],
    )

    print(
        "Fusion Confidence:",
        f"{result['confidence']:.4f}",
    )

    print(
        "LSTM Prediction  :",
        result["lstm_prediction"],
    )

    print(
        "LSTM Confidence  :",
        f"{result['lstm_confidence']:.4f}",
    )

    print(
        "GRU Prediction   :",
        result["gru_prediction"],
    )

    print(
        "GRU Confidence   :",
        f"{result['gru_confidence']:.4f}",
    )

    print(
        "Models Agree     :",
        result["agreement"],
    )

    print()

    print("=" * 60)
    print("FUSION PIPELINE TEST COMPLETED")
    print("=" * 60)