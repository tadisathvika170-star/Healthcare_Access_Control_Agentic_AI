"""
fusion.py

Fusion module that combines predictions from
the LSTM and GRU models.

This version returns a complete dictionary that is
compatible with the Agentic AI pipeline.
"""

from predict.predict_lstm import LSTMPredictor
from predict.predict_gru import GRUPredictor


class FusionPredictor:
    """
    Combines predictions from LSTM and GRU models.
    """

    def __init__(self):

        self.lstm = LSTMPredictor()
        self.gru = GRUPredictor()

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self, features):

        # ---------------------------------------------
        # Individual Model Predictions
        # ---------------------------------------------

        lstm_prediction, lstm_confidence = self.lstm.predict(features)

        gru_prediction, gru_confidence = self.gru.predict(features)

        # ---------------------------------------------
        # Fusion Logic
        # ---------------------------------------------

        if lstm_prediction == gru_prediction:

            predicted_user = lstm_prediction

            confidence = (
                lstm_confidence +
                gru_confidence
            ) / 2

            agreement = True

        else:

            agreement = False

            if lstm_confidence >= gru_confidence:

                predicted_user = lstm_prediction

                confidence = lstm_confidence * 0.90

            else:

                predicted_user = gru_prediction

                confidence = gru_confidence * 0.90

        # ---------------------------------------------
        # Return Dictionary
        # ---------------------------------------------

        return {

            "predicted_user": predicted_user,

            "confidence": round(
                confidence,
                4
            ),

            "lstm_prediction": lstm_prediction,

            "gru_prediction": gru_prediction,

            "lstm_confidence": round(
                lstm_confidence,
                4
            ),

            "gru_confidence": round(
                gru_confidence,
                4
            ),

            "agreement": agreement

        }


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    import numpy as np

    fusion = FusionPredictor()

    sample = np.random.rand(33)

    result = fusion.predict(sample)

    print("=" * 60)
    print("Fusion Predictor Test")
    print("=" * 60)

    for key, value in result.items():

        print(f"{key:20}: {value}")