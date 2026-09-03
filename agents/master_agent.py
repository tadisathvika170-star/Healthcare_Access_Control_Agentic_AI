"""
master_agent.py

Master Agent

Coordinates the complete Agentic AI authentication workflow.
"""

from agents.memory import AgentMemory


class MasterAgent:

    def __init__(self):

        self.memory = AgentMemory()

        self.credential_agent = None
        self.identity_agent = None
        self.confidence_agent = None
        self.risk_agent = None
        self.strategy_agent = None
        self.decision_agent = None
        self.audit_agent = None
        self.logger_agent = None

    # ==========================================================
    # Register Agents
    # ==========================================================

    def register_agents(
        self,
        credential_agent,
        identity_agent,
        confidence_agent,
        risk_agent,
        strategy_agent,
        decision_agent,
        audit_agent,
        logger_agent,
    ):

        self.credential_agent = credential_agent
        self.identity_agent = identity_agent
        self.confidence_agent = confidence_agent
        self.risk_agent = risk_agent
        self.strategy_agent = strategy_agent
        self.decision_agent = decision_agent
        self.audit_agent = audit_agent
        self.logger_agent = logger_agent

    # ==========================================================
    # Authentication Pipeline
    # ==========================================================

    def authenticate(
        self,
        username,
        password,
        features
    ):

        self.memory.clear()

        try:

            # --------------------------------------------------
            # STEP 1 : Credential Verification
            # --------------------------------------------------

            credential_result = self.credential_agent.authenticate(
                username,
                password
            )

            self.memory.set(
                "credential_result",
                credential_result
            )

            if not credential_result["success"]:

                if self.logger_agent:

                    self.logger_agent.warning(
                        f"Credential verification failed for {username}"
                    )

                return credential_result

            # --------------------------------------------------
            # STEP 2 : Identity Verification
            # --------------------------------------------------

            identity_result = self.identity_agent.verify_identity(
                features
            )

            self.memory.set(
                "identity_result",
                identity_result
            )

            # --------------------------------------------------
            # STEP 3 : Confidence Evaluation
            # --------------------------------------------------

            confidence_result = self.confidence_agent.calculate(
                identity_result["confidence"]
            )

            self.memory.set(
                "confidence_result",
                confidence_result
            )

            # --------------------------------------------------
            # STEP 4 : Model Agreement
            # --------------------------------------------------

            model_agreement = (
                identity_result["lstm_prediction"] ==
                identity_result["gru_prediction"]
            )

            self.memory.set(
                "model_agreement",
                model_agreement
            )

            # --------------------------------------------------
            # STEP 5 : Risk Assessment
            # --------------------------------------------------

            risk_result = self.risk_agent.evaluate(

                confidence=confidence_result["confidence"],

                model_agreement=model_agreement,

                failed_attempts=0,

                trusted_user=True

            )

            self.memory.set(
                "risk_result",
                risk_result
            )

            # --------------------------------------------------
            # STEP 6 : Strategy Selection
            # --------------------------------------------------

            strategy_result = self.strategy_agent.choose_strategy(
                risk_result
            )

            self.memory.set(
                "strategy_result",
                strategy_result
            )

            # --------------------------------------------------
            # STEP 7 : Final Decision
            # --------------------------------------------------

            decision = self.decision_agent.make_decision(

                credential_result,

                identity_result,

                confidence_result,

                risk_result,

                strategy_result

            )

            self.memory.set(
                "decision",
                decision
            )

            # --------------------------------------------------
            # STEP 8 : Audit
            # --------------------------------------------------

            if self.audit_agent:

                self.audit_agent.log_event(

                    username=username,

                    action="LOGIN",

                    status=decision["status"],

                    confidence=decision.get(
                        "confidence",
                        0
                    ),

                    risk_level=decision.get(
                        "risk_level",
                        "UNKNOWN"
                    )

                )

            # --------------------------------------------------
            # STEP 9 : Logging
            # --------------------------------------------------

            if self.logger_agent:

                self.logger_agent.info(

                    f"""
Authentication Summary

User           : {username}
Risk           : {risk_result['risk_level']}
Risk Score     : {risk_result['risk_score']}
Action         : {decision['action']}
Authenticated  : {decision['authenticated']}
"""

                )

            # --------------------------------------------------
            # STEP 10 : Store Session Memory
            # --------------------------------------------------

            self.memory.set(

                "authentication_summary",

                {

                    "username": username,

                    "confidence":
                        confidence_result["confidence"],

                    "risk":
                        risk_result["risk_level"],

                    "decision":
                        decision["action"]

                }

            )

            return decision

        except Exception as error:

            if self.logger_agent:

                self.logger_agent.error(str(error))

            return {

                "authenticated": False,

                "status": "FAILED",

                "action": "SYSTEM_ERROR",

                "message": str(error)

            }

    # ==========================================================
    # Memory
    # ==========================================================

    def get_memory(self):

        return self.memory.all()

    def clear_memory(self):

        self.memory.clear()


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("MASTER AGENT READY")

    print("=" * 60)