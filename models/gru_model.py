"""
gru_model.py

Feature-Enhanced GRU model for
3-repetition keystroke authentication.
"""

import torch
import torch.nn as nn


class FeatureEnhancedGRU(nn.Module):

    def __init__(
        self,
        input_size=31,
        hidden_size=128,
        num_layers=1,
        num_classes=51,
        dropout=0.0,
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        # =====================================================
        # GRU BRANCH
        # =====================================================

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )


        # =====================================================
        # STATISTICAL FEATURE BRANCH
        # =====================================================

        statistical_size = (
            input_size * 5
        )

        self.statistics = nn.Sequential(

            nn.Linear(
                statistical_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU()
        )


        # =====================================================
        # FUSION
        # =====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size + 64,
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

        # =====================================================
        # GRU FEATURES
        # =====================================================

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )

        gru_output, _ = self.gru(
            x,
            h0
        )

        gru_features = gru_output[
            :, -1, :
        ]


        # =====================================================
        # STATISTICAL FEATURES
        # =====================================================

        mean_features = x.mean(
            dim=1
        )

        std_features = x.std(
            dim=1,
            unbiased=False
        )

        min_features = x.min(
            dim=1
        ).values

        max_features = x.max(
            dim=1
        ).values

        range_features = (
            max_features -
            min_features
        )


        statistical_features = torch.cat(
            [
                mean_features,
                std_features,
                min_features,
                max_features,
                range_features,
            ],
            dim=1
        )


        statistical_features = (
            self.statistics(
                statistical_features
            )
        )


        # =====================================================
        # FUSION
        # =====================================================

        fused = torch.cat(
            [
                gru_features,
                statistical_features,
            ],
            dim=1
        )


        return self.classifier(
            fused
        )


# Backward compatibility.
GRUModel = FeatureEnhancedGRU


if __name__ == "__main__":

    model = FeatureEnhancedGRU()

    sample = torch.randn(
        8,
        3,
        31
    )

    prediction = model(
        sample
    )

    print("=" * 60)
    print("FEATURE-ENHANCED GRU TEST")
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