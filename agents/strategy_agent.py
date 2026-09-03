"""
strategy_agent.py

Adaptive Strategy Agent

Determines the next authentication action based on
the evaluated risk level and risk score.
"""


class StrategyAgent:
    """
    Adaptive Authentication Strategy Agent
    """

    def __init__(self):
        pass

    # ==========================================================
    # Strategy Selection
    # ==========================================================

    def choose_strategy(self, risk_result):

        risk_level = risk_result.get("risk_level", "UNKNOWN")
        risk_score = risk_result.get("risk_score", 0)

        # ======================================================
        # LOW RISK
        # ======================================================

        if risk_level == "LOW":

            return {

                "action": "GRANT_ACCESS",

                "message": "Authentication successful.",

                "next_step":
                    "Allow user to access the healthcare system.",

                "monitoring":
                    "Normal monitoring.",

                "requires_admin":
                    False,

                "session_timeout":
                    30,

                "risk_score":
                    risk_score
            }

        # ======================================================
        # MEDIUM RISK
        # ======================================================

        elif risk_level == "MEDIUM":

            return {

                "action": "REAUTHENTICATE",

                "message":
                    "Please complete one more authentication step.",

                "next_step":
                    "Request another keystroke sample.",

                "monitoring":
                    "Increase monitoring for this session.",

                "requires_admin":
                    False,

                "session_timeout":
                    15,

                "risk_score":
                    risk_score
            }

        # ======================================================
        # HIGH RISK
        # ======================================================

        elif risk_level == "HIGH":

            return {

                "action": "DENY_ACCESS",

                "message":
                    "Authentication denied due to high risk.",

                "next_step":
                    "Reject login attempt and log the event.",

                "monitoring":
                    "Flag account for review.",

                "requires_admin":
                    True,

                "session_timeout":
                    0,

                "risk_score":
                    risk_score
            }

        # ======================================================
        # CRITICAL RISK
        # ======================================================

        else:

            return {

                "action": "LOCK_ACCOUNT",

                "message":
                    "Critical security risk detected.",

                "next_step":
                    "Temporarily lock the account.",

                "monitoring":
                    "Immediately notify administrator.",

                "requires_admin":
                    True,

                "session_timeout":
                    0,

                "risk_score":
                    risk_score
            }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    agent = StrategyAgent()

    sample = {

        "risk_level": "LOW",

        "risk_score": 10

    }

    print(agent.choose_strategy(sample))