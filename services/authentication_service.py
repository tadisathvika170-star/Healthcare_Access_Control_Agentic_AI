"""
authentication_service.py

Authentication Service

Normal login:
    Username + Password
        ↓
    Login successful

Suspicious login:
    Username + Password
        ↓
    Doubt detected
        ↓
    Keystroke verification
"""

from authentication.auth_manager import AuthManager
from predict.predictor import Predictor


class AuthenticationService:

    """
    Handles authentication.
    Keystroke verification is performed only
    when additional verification is required.
    """

    def __init__(self):

        self.auth_manager = AuthManager()

        self.predictor = Predictor()

    # ======================================================
    # Normal Username + Password Login
    # ======================================================

    def login(
        self,
        username,
        password
    ):

        success, result = (
            self.auth_manager.login_user(
                username,
                password
            )
        )

        if success:

            return {

                "success": True,

                "message":
                    "Username and password verified.",

                "user":
                    result

            }

        return {

            "success": False,

            "message":
                result,

            "user":
                None

        }

    # ======================================================
    # Check Login Doubt
    # ======================================================

    def check_login_doubt(
        self,
        failed_attempts=0
    ):

        risk_score = 0

        reasons = []

        # --------------------------------------------------
        # Failed Attempts
        # --------------------------------------------------

        if failed_attempts >= 3:

            risk_score += 2

            reasons.append(
                "Multiple failed login attempts."
            )

        elif failed_attempts >= 1:

            risk_score += 1

            reasons.append(
                "Previous failed login attempt."
            )

        # --------------------------------------------------
        # Decide
        # --------------------------------------------------

        require_keystroke = (
            risk_score >= 2
        )

        return {

            "require_keystroke":
                require_keystroke,

            "risk_score":
                risk_score,

            "reasons":
                reasons

        }

    # ======================================================
    # Keystroke Verification
    # ======================================================

    def verify_keystroke(
        self,
        username,
        features
    ):

        result = self.predictor.predict(
            features
        )

        predicted_user = result[
            "predicted_user"
        ]

        authenticated = (
            predicted_user == username
        )

        return {

            "authenticated":
                authenticated,

            "expected_user":
                username,

            "predicted_user":
                predicted_user,

            "confidence":
                result["confidence"],

            "agreement":
                result["agreement"],

            "lstm_confidence":
                result["lstm_confidence"],

            "gru_confidence":
                result["gru_confidence"]

        }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    service = AuthenticationService()

    result = service.login(
        "s041",
        "password"
    )

    print(result)

    doubt = service.check_login_doubt(
        failed_attempts=3
    )

    print(doubt)