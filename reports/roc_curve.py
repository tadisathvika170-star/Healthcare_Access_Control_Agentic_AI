"""
roc_curve.py

Plots ROC curves for the trained
LSTM and GRU models.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataset.dataset import KeystrokeDataset
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

from config import (
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_models(num_classes):

    lstm = LSTMModel(
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
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
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)

    gru.load_state_dict(
        torch.load(
            GRU_MODEL_PATH,
            map_location=device,
        )
    )

    return lstm, gru


def collect_scores(model, loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for inputs, target in loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)

            probabilities.append(
                probs.cpu().numpy()
            )

            labels.append(
                target.numpy()
            )

    probabilities = np.vstack(probabilities)

    labels = np.concatenate(labels)

    return labels, probabilities


def plot_macro_roc(labels, probabilities, title):

    classes = np.unique(labels)

    y_true = label_binarize(labels, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):

        fpr[i], tpr[i], _ = roc_curve(
            y_true[:, i],
            probabilities[:, i]
        )

        roc_auc[i] = auc(
            fpr[i],
            tpr[i]
        )

    plt.figure(figsize=(7,6))

    for i in range(len(classes)):

        plt.plot(
            fpr[i],
            tpr[i],
            lw=1,
            alpha=0.4
        )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(title)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


def main():

    dataset = KeystrokeDataset()

    loader = dataset.test_loader()

    num_classes = len(torch.unique(dataset.y_train))

    lstm, gru = load_models(num_classes)

    labels, lstm_prob = collect_scores(
        lstm,
        loader
    )

    _, gru_prob = collect_scores(
        gru,
        loader
    )

    plot_macro_roc(
        labels,
        lstm_prob,
        "LSTM ROC Curves"
    )

    plot_macro_roc(
        labels,
        gru_prob,
        "GRU ROC Curves"
    )


if __name__ == "__main__":

    main()