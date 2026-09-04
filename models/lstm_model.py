"""
lstm_model.py

LSTM model for 3-repetition keystroke
behavioral biometric authentication.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=31,
        hidden_size=64,
        num_layers=2,
        num_classes=51,
        dropout=0.2,
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )

        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )

        output, _ = self.lstm(
            x,
            (h0, c0)
        )

        output = output[:, -1, :]

        output = self.classifier(
            output
        )

        return output


if __name__ == "__main__":

    model = LSTMModel()

    sample = torch.randn(
        8,
        3,
        31
    )

    prediction = model(sample)

    print("=" * 60)
    print("LSTM MODEL TEST")
    print("=" * 60)
    print(
        "Input shape :",
        sample.shape
    )
    print(
        "Output shape:",
        prediction.shape
    )
    print("=" * 60)