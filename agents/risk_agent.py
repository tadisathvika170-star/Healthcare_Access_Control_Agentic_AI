"""
risk_agent.py

Advanced Risk Assessment Agent
"""


class RiskAgent:
    """
    Evaluates authentication risk using multiple factors.
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        confidence: float,
        model_agreement: bool = True,
        failed_attempts: int = 0,
        trusted_user: bool = True,
    ):

        reasons = []

        score = 0

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        if confidence >= 0.95:

            score += 4
            reasons.append("Very high confidence score.")

        elif confidence >= 0.90:

            score += 3
            reasons.append("High confidence score.")

        elif confidence >= 0.80:

            score += 2
            reasons.append("Moderate confidence score.")

        elif confidence >= 0.70:

            score += 1
            reasons.append("Low confidence score.")

        else:

            reasons.append("Very low confidence score.")

        # -----------------------------------------
        # Model Agreement
        # -----------------------------------------

        if model_agreement:

            score += 2
            reasons.append("LSTM and GRU predictions agree.")

        else:

            reasons.append("LSTM and GRU predictions disagree.")

        # -----------------------------------------
        # Failed Attempts
        # -----------------------------------------

        if failed_attempts == 0:

            score += 2
            reasons.append("No recent failed login attempts.")

        elif failed_attempts <= 2:

            score += 1
            reasons.append("Few failed login attempts.")

        else:

            reasons.append("Multiple failed login attempts detected.")

        # -----------------------------------------
        # Trusted User
        # -----------------------------------------

        if trusted_user:

            score += 2
            reasons.append("Trusted user.")

        else:

            reasons.append("Unknown or untrusted user.")

        # -----------------------------------------
        # Final Risk
        # -----------------------------------------

        if score >= 9:

            risk = "LOW"

        elif score >= 7:

            risk = "MEDIUM"

        elif score >= 5:

            risk = "HIGH"

        else:

            risk = "CRITICAL"

        return {

            "risk_level": risk,

            "risk_score": score,

            "confidence": round(confidence, 4),

            "reasons": reasons

        }


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    agent = RiskAgent()

    result = agent.evaluate(
        confidence=0.97,
        model_agreement=True,
        failed_attempts=0,
        trusted_user=True
    )

    print(result)