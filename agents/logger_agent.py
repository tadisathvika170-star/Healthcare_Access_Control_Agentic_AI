"""
logger_agent.py

Professional Logger Agent

Handles logging for authentication events,
AI decisions, warnings, and system errors.
"""

import logging
import os

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


class LoggerAgent:

    def __init__(self):

        self.logger = logging.getLogger("HealthcareAgent")

        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            formatter = logging.Formatter(

                "%(asctime)s | %(levelname)s | %(message)s"

            )

            # -------------------------------------------------
            # System Log
            # -------------------------------------------------

            system_handler = logging.FileHandler(

                os.path.join(LOG_DIR, "system.log")

            )

            system_handler.setFormatter(formatter)

            self.logger.addHandler(system_handler)

            # -------------------------------------------------
            # Error Log
            # -------------------------------------------------

            error_handler = logging.FileHandler(

                os.path.join(LOG_DIR, "error.log")

            )

            error_handler.setLevel(logging.ERROR)

            error_handler.setFormatter(formatter)

            self.logger.addHandler(error_handler)

    # =====================================================
    # INFO
    # =====================================================

    def info(self, message):

        self.logger.info(message)

    # =====================================================
    # WARNING
    # =====================================================

    def warning(self, message):

        self.logger.warning(message)

    # =====================================================
    # ERROR
    # =====================================================

    def error(self, message):

        self.logger.error(message)

    # =====================================================
    # Authentication Log
    # =====================================================

    def log_authentication(

        self,

        username,

        decision,

        confidence,

        risk

    ):

        self.logger.info(

            f"""
Authentication Event

User        : {username}
Decision    : {decision}
Confidence  : {confidence:.2f}
Risk Level  : {risk}
"""
        )

    # =====================================================
    # Agent Decision Log
    # =====================================================

    def log_agent_reasoning(

        self,

        reasoning

    ):

        self.logger.info(

            "AI Reasoning:\n" +

            "\n".join(reasoning)

        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    logger = LoggerAgent()

    logger.info("Logger Agent Started")

    logger.warning("Sample Warning")

    logger.error("Sample Error")

    logger.log_authentication(

        username="doctor",

        decision="GRANT_ACCESS",

        confidence=0.97,

        risk="LOW"

    )

    logger.log_agent_reasoning(

        [

            "Credentials verified",

            "LSTM and GRU agree",

            "Low risk"

        ]

    )

    print("Logger Agent Ready")