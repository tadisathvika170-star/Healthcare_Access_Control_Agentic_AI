import numpy as np
import torch

from models.lstm_model import LSTMModel


INPUT_SIZE = 31
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_CLASSES = 51
SEQUENCE_LENGTH = 10

MODEL_PATH = "models/lstm_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=0.0
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


def predict_lstm(sequence):
    """
    Predict user from a 10 x 31 keystroke sequence.

    Parameters
    ----------
    sequence : numpy array
        Shape: (10, 31)

    Returns
    -------
    dict
        Predicted class and confidence.
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        INPUT_SIZE
    ):
        raise ValueError(
            f"Expected sequence shape "
            f"(10, 31), got {sequence.shape}"
        )

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "predicted_class": prediction.item(),
        "confidence": confidence.item()
    }