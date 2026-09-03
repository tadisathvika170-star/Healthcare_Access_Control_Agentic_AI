"""
identity_agent.py

Identity Agent

Keystroke verification is used only when
the login system has doubt about the user.
"""

from predict.predictor import Predictor


class IdentityAgent:
    """
    Identity Verification Agent
    """

    def __init__(self):

        self.predictor = Predictor()

    # ======================================================
    # Check Whether Additional Verification Is Required
    # ======================================================

    def check_login_doubt(
        self,
        failed_attempts=0,
        new_login=True
    ):
        """
        Simple risk check before keystroke authentication.

        Keystroke verification is requested only when
        there is some doubt about the login.

        Rules:
        - Multiple failed attempts -> doubt
        - New login after suspicious attempts -> doubt
        """

        risk_score = 0
        reasons = []

        # --------------------------------------------------
        # Failed Login Attempts
        # --------------------------------------------------

        if failed_attempts >= 3:

            risk_score += 2

            reasons.append(
                "Multiple failed login attempts detected."
            )

        elif failed_attempts >= 1:

            risk_score += 1

            reasons.append(
                "Previous failed login attempt detected."
            )

        # --------------------------------------------------
        # New Login
        # --------------------------------------------------

        if new_login:

            risk_score += 1

            reasons.append(
                "New login session detected."
            )

        # --------------------------------------------------
        # Final Decision
        # --------------------------------------------------

        require_keystroke = risk_score >= 2

        return {

            "require_keystroke": require_keystroke,

            "risk_score": risk_score,

            "reasons": reasons

        }

    # ======================================================
    # Keystroke Identity Verification
    # ======================================================

    def verify_identity(self, features):

        """
        Verify user identity using keystroke behavior.
        """

        prediction = self.predictor.predict(
            features
        )

        return {

            "success": True,

            "predicted_user":
                prediction["predicted_user"],

            "confidence":
                prediction["confidence"],

            "lstm_prediction":
                prediction["lstm_prediction"],

            "gru_prediction":
                prediction["gru_prediction"]

        }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    agent = IdentityAgent()

    result = agent.check_login_doubt(
        failed_attempts=3,
        new_login=True
    )

    print(result)