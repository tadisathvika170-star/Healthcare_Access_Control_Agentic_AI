"""
LSTM + Feature-Enhanced GRU fusion for single-session keystroke verification.

The trained models accept variable sequence lengths. The application now
uses one completed phrase for one biometric verification event.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
LSTM_PATH = MODEL_DIR / "lstm_model.pth"
GRU_PATH = MODEL_DIR / "gru_model.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

SEQUENCE_LENGTH = 1
INPUT_SIZE = 31
NUM_CLASSES = 51


class LSTMModel(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=2, num_classes=51):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


class FeatureEnhancedGRU(nn.Module):
    def __init__(self, input_size=31, hidden_size=128, num_layers=1, num_classes=51):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.statistics = nn.Sequential(
            nn.Linear(input_size * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        gru_output, _ = self.gru(x)
        gru_features = gru_output[:, -1, :]

        mean_features = torch.mean(x, dim=1)
        std_features = torch.std(x, dim=1, unbiased=False)
        min_features = torch.min(x, dim=1).values
        max_features = torch.max(x, dim=1).values
        range_features = max_features - min_features

        statistical_features = torch.cat(
            [mean_features, std_features, min_features, max_features, range_features],
            dim=1,
        )
        statistical_features = self.statistics(statistical_features)

        return self.classifier(torch.cat([gru_features, statistical_features], dim=1))


class FusionPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for path in (LSTM_PATH, GRU_PATH, SCALER_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        self.lstm_model = LSTMModel()
        self.lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=self.device))
        self.lstm_model.to(self.device).eval()

        self.gru_model = FeatureEnhancedGRU()
        self.gru_model.load_state_dict(torch.load(GRU_PATH, map_location=self.device))
        self.gru_model.to(self.device).eval()

        self.scaler = joblib.load(SCALER_PATH)

    def _prepare_sequence(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(
                f"Invalid keystroke sequence shape. Expected {(SEQUENCE_LENGTH, INPUT_SIZE)}; received {sequence.shape}."
            )

        scaled_sequence = self.scaler.transform(sequence)
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, sequence):
        x = self._prepare_sequence(sequence)

        lstm_logits = self.lstm_model(x)
        lstm_probabilities = torch.softmax(lstm_logits, dim=1)
        lstm_confidence, lstm_prediction = torch.max(lstm_probabilities, dim=1)

        gru_logits = self.gru_model(x)
        gru_probabilities = torch.softmax(gru_logits, dim=1)
        gru_confidence, gru_prediction = torch.max(gru_probabilities, dim=1)

        fusion_probabilities = (lstm_probabilities + gru_probabilities) / 2.0
        fusion_confidence, fusion_prediction = torch.max(fusion_probabilities, dim=1)

        return {
            "predicted_user": int(fusion_prediction.item()),
            "confidence": float(fusion_confidence.item()),
            "lstm_prediction": int(lstm_prediction.item()),
            "lstm_confidence": float(lstm_confidence.item()),
            "gru_prediction": int(gru_prediction.item()),
            "gru_confidence": float(gru_confidence.item()),
            "agreement": bool(lstm_prediction.item() == gru_prediction.item()),
        }


def predict_fusion(sequence):
    return FusionPredictor().predict(sequence)
