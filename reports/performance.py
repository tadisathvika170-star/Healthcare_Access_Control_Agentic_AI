"""
performance.py

Displays a comparison of Accuracy,
Precision, Recall and F1 Score
between LSTM and GRU models.
"""

import matplotlib.pyplot as plt
import numpy as np
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

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    lstm_scores = [
        lstm_result["accuracy"],
        lstm_result["precision"],
        lstm_result["recall"],
        lstm_result["f1_score"],
    ]

    gru_scores = [
        gru_result["accuracy"],
        gru_result["precision"],
        gru_result["recall"],
        gru_result["f1_score"],
    ]

    x = np.arange(len(metrics))

    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width/2,
        lstm_scores,
        width,
        label="LSTM"
    )

    plt.bar(
        x + width/2,
        gru_scores,
        width,
        label="GRU"
    )

    plt.xticks(x, metrics)

    plt.ylim(0, 1.05)

    plt.ylabel("Score")

    plt.title("Performance Comparison")

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    main()