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

        Expected input:
            (10, 31)

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

        result = self.fusion.predict(
            features
        )

        return result


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    import numpy as np

    predictor = Predictor()

    # --------------------------------------------------
    # Test with the correct trained-model shape.
    #
    # The production models require:
    #   10 time steps
    #   31 features per time step
    # --------------------------------------------------

    sample = np.random.rand(
        10,
        31
    ).astype(
        np.float32
    )

    result = predictor.predict(
        sample
    )

    print("=" * 60)
    print("Predictor Test")
    print("=" * 60)

    print(
        f"Predicted User   : "
        f"{result['predicted_user']}"
    )

    print(
        f"Confidence       : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"LSTM Prediction  : "
        f"{result['lstm_prediction']}"
    )

    print(
        f"GRU Prediction   : "
        f"{result['gru_prediction']}"
    )

    print(
        f"LSTM Confidence  : "
        f"{result['lstm_confidence']:.4f}"
    )

    print(
        f"GRU Confidence   : "
        f"{result['gru_confidence']:.4f}"
    )

    print(
        f"Models Agree     : "
        f"{result['agreement']}"
    )

    print("=" * 60)