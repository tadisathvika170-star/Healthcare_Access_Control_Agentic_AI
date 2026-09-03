"""
predictor.py

Main prediction interface for the Agentic AI
Healthcare Access Control System.
"""

from predict.fusion import FusionPredictor


class Predictor:
    """
    Main prediction interface.

    Used by:
    - Identity Agent
    - Authentication Service
    - Master Agent
    """

    def __init__(self):

        self.fusion = FusionPredictor()

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self, features):
        """
        Predict user from keystroke features.

        Returns
        -------
        dict
            {
                predicted_user,
                confidence,
                lstm_prediction,
                gru_prediction,
                lstm_confidence,
                gru_confidence,
                agreement
            }
        """

        result = self.fusion.predict(features)

        return result


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    import numpy as np

    predictor = Predictor()

    sample = np.random.rand(33)

    result = predictor.predict(sample)

    print("=" * 60)
    print("Predictor Test")
    print("=" * 60)

    print(f"Predicted User   : {result['predicted_user']}")
    print(f"Confidence       : {result['confidence']:.4f}")
    print(f"LSTM Prediction  : {result['lstm_prediction']}")
    print(f"GRU Prediction   : {result['gru_prediction']}")
    print(f"LSTM Confidence  : {result['lstm_confidence']:.4f}")
    print(f"GRU Confidence   : {result['gru_confidence']:.4f}")
    print(f"Models Agree     : {result['agreement']}")