"""
Confidence Agent

This agent calculates the confidence score
for the predicted identity based on the
outputs of the LSTM and GRU models.
"""

from config import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    LOW_CONFIDENCE
)


class ConfidenceAgent:
    """
    Calculates confidence and assigns a confidence level.
    """

    def calculate(self, confidence: float):
        """
        Parameters
        ----------
        confidence : float
            Confidence score between 0 and 1.

        Returns
        -------
        dict
        """

        if confidence >= HIGH_CONFIDENCE:
            level = "HIGH"

        elif confidence >= MEDIUM_CONFIDENCE:
            level = "MEDIUM"

        elif confidence >= LOW_CONFIDENCE:
            level = "LOW"

        else:
            level = "VERY LOW"

        return {
            "confidence": round(confidence, 4),
            "confidence_level": level
        }