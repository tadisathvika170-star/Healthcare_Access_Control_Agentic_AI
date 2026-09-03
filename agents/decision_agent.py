"""
decision_agent.py

Decision Agent

Decides whether the user can login normally
or whether additional keystroke verification
is required.
"""


class DecisionAgent:

    """
    Final Authentication Decision Agent.
    """

    def __init__(self):
        pass

    # ======================================================
    # Decide Login Action
    # ======================================================

    def decide_login(
        self,
        credential_result,
        doubt_result
    ):

        reasoning = []

        # --------------------------------------------------
        # Credential Check
        # --------------------------------------------------

        if not credential_result["success"]:

            reasoning.append(
                "Username or password is incorrect."
            )

            return {

                "authenticated": False,

                "status": "FAILED",

                "action": "DENY_ACCESS",

                "require_keystroke": False,

                "message":
                    credential_result["message"],

                "reasoning":
                    reasoning

            }

        reasoning.append(
            "Username and password verified successfully."
        )

        # --------------------------------------------------
        # Doubt Check
        # --------------------------------------------------

        if doubt_result["require_keystroke"]:

            reasoning.append(
                "The system detected some doubt about "
                "the current login."
            )

            reasoning.extend(
                doubt_result["reasons"]
            )

            reasoning.append(
                "Additional keystroke verification is required."
            )

            return {

                "authenticated": False,

                "status": "ADDITIONAL_VERIFICATION",

                "action": "REQUIRE_KEYSTROKE",

                "require_keystroke": True,

                "message":
                    "Additional verification required.",

                "reasoning":
                    reasoning,

                "risk_score":
                    doubt_result["risk_score"]

            }

        # --------------------------------------------------
        # Normal Login
        # --------------------------------------------------

        reasoning.append(
            "No significant doubt detected."
        )

        reasoning.append(
            "Keystroke verification is not required."
        )

        return {

            "authenticated": True,

            "status": "SUCCESS",

            "action": "GRANT_ACCESS",

            "require_keystroke": False,

            "message":
                "Login successful.",

            "reasoning":
                reasoning,

            "risk_score":
                doubt_result["risk_score"]

        }


# ==========================================================
# Existing Full Decision Function
# ==========================================================

    def make_decision(
        self,
        credential_result,
        identity_result,
        confidence_result,
        risk_result,
        strategy_result
    ):

        reasoning = []

        # --------------------------------------------------
        # Credentials
        # --------------------------------------------------

        if credential_result["success"]:

            reasoning.append(
                "Username and password verified successfully."
            )

        else:

            return {

                "authenticated": False,

                "status": "FAILED",

                "action": "DENY_ACCESS",

                "message":
                    credential_result["message"],

                "reasoning": [
                    "Credential verification failed."
                ],

                "trust_score": 0.0

            }

        # --------------------------------------------------
        # Identity
        # --------------------------------------------------

        if identity_result["success"]:

            reasoning.append(
                "Behavioral biometric verification completed."
            )

        else:

            return {

                "authenticated": False,

                "status": "FAILED",

                "action": "DENY_ACCESS",

                "message":
                    "Identity verification failed.",

                "reasoning":
                    reasoning,

                "trust_score": 0.0

            }

        # --------------------------------------------------
        # Model Agreement
        # --------------------------------------------------

        lstm_prediction = identity_result.get(
            "lstm_prediction"
        )

        gru_prediction = identity_result.get(
            "gru_prediction"
        )

        if lstm_prediction == gru_prediction:

            reasoning.append(
                "LSTM and GRU predictions agree."
            )

        else:

            reasoning.append(
                "LSTM and GRU predictions differ."
            )

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence = confidence_result["confidence"]

        reasoning.append(
            f"Confidence Score: {confidence * 100:.2f}%"
        )

        reasoning.append(
            f"Confidence Level: "
            f"{confidence_result['confidence_level']}"
        )

        # --------------------------------------------------
        # Risk
        # --------------------------------------------------

        reasoning.append(
            f"Risk Level: {risk_result['risk_level']}"
        )

        # --------------------------------------------------
        # Final Decision
        # --------------------------------------------------

        authenticated = (
            strategy_result["action"]
            == "GRANT_ACCESS"
        )

        return {

            "authenticated":
                authenticated,

            "status":
                "SUCCESS"
                if authenticated
                else "FAILED",

            "action":
                strategy_result["action"],

            "message":
                strategy_result["message"],

            "predicted_user":
                identity_result["predicted_user"],

            "confidence":
                confidence,

            "confidence_level":
                confidence_result["confidence_level"],

            "risk_level":
                risk_result["risk_level"],

            "trust_score":
                round(confidence * 100, 2),

            "reasoning":
                reasoning

        }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    agent = DecisionAgent()

    credential = {
        "success": True,
        "message": "Credentials valid."
    }

    doubt = {
        "require_keystroke": True,
        "risk_score": 2,
        "reasons": [
            "Suspicious login detected."
        ]
    }

    print(
        agent.decide_login(
            credential,
            doubt
        )
    )