"""
evaluate.py

Evaluate trained LSTM and GRU models.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
)

from dataset.dataset import KeystrokeDataset
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ModelEvaluator:

    def __init__(self):

        self.dataset = KeystrokeDataset(batch_size=32)

        self.test_loader = self.dataset.test_loader()

        self.num_classes = len(torch.unique(self.dataset.y_train))

    def evaluate(self, model, model_name):

        print("\n" + "=" * 60)
        print(f"Evaluating {model_name}")
        print("=" * 60)

        model.eval()

        predictions = []
        actual = []

        with torch.no_grad():

            for inputs, labels in self.test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                actual.extend(labels.cpu().numpy())

        accuracy = accuracy_score(actual, predictions)

        precision = precision_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(actual, predictions)

        report = classification_report(
            actual,
            predictions,
            zero_division=0,
        )

        print(f"\nAccuracy : {accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix:\n")
        print(cm)

        print("\nClassification Report:\n")
        print(report)

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "actual": actual,
            "predictions": predictions,
        }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    evaluator = ModelEvaluator()

    # ------------------------------------------------------
    # LSTM
    # ------------------------------------------------------

    lstm_model = LSTMModel(
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=evaluator.num_classes,
        dropout=0.2,
    ).to(device)

    lstm_model.load_state_dict(
        torch.load(
            LSTM_MODEL_PATH,
            map_location=device,
        )
    )

    lstm_results = evaluator.evaluate(
        lstm_model,
        "LSTM Model",
    )

    # ------------------------------------------------------
    # GRU
    # ------------------------------------------------------

    gru_model = GRUModel(
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=evaluator.num_classes,
        dropout=0.2,
    ).to(device)

    gru_model.load_state_dict(
        torch.load(
            GRU_MODEL_PATH,
            map_location=device,
        )
    )

    gru_results = evaluator.evaluate(
        gru_model,
        "GRU Model",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"LSTM Accuracy : {lstm_results['accuracy']*100:.2f}%")
    print(f"GRU Accuracy  : {gru_results['accuracy']*100:.2f}%")

    print("\nMODEL EVALUATION COMPLETED")