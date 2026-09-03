"""
confusion_matrix.py

Displays confusion matrices for
both LSTM and GRU models.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch

from models.evaluate import ModelEvaluator
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

from config import (
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_models(evaluator):

    lstm = LSTMModel(
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=evaluator.num_classes,
        dropout=0.2,
    ).to(device)

    lstm.load_state_dict(
        torch.load(
            LSTM_MODEL_PATH,
            map_location=device,
        )
    )

    gru = GRUModel(
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=evaluator.num_classes,
        dropout=0.2,
    ).to(device)

    gru.load_state_dict(
        torch.load(
            GRU_MODEL_PATH,
            map_location=device,
        )
    )

    return lstm, gru


def plot_matrix(result, title):

    plt.figure(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"]
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(title)

    plt.tight_layout()

    plt.show()


def main():

    evaluator = ModelEvaluator()

    lstm, gru = load_models(evaluator)

    lstm_result = evaluator.evaluate(
        lstm,
        "LSTM"
    )

    gru_result = evaluator.evaluate(
        gru,
        "GRU"
    )

    plot_matrix(
        lstm_result,
        "LSTM Confusion Matrix"
    )

    plot_matrix(
        gru_result,
        "GRU Confusion Matrix"
    )


if __name__ == "__main__":

    main()