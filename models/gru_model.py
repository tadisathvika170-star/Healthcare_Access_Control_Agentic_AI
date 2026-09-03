"""
gru_model.py

GRU model for keystroke dynamics authentication.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU Network for User Authentication
    """

    def __init__(
        self,
        input_size=33,
        hidden_size=64,
        num_layers=2,
        num_classes=51,
        dropout=0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # Initial hidden state
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        output, _ = self.gru(x, h0)

        output = output[:, -1, :]

        output = self.classifier(output)

        return output


if __name__ == "__main__":

    model = GRUModel()

    print(model)

    sample = torch.randn(8, 1, 33)

    prediction = model(sample)

    print()

    print("Input Shape :", sample.shape)
    print("Output Shape:", prediction.shape)