"""
logger.py

Advanced logging utility for the
Healthcare Access Control Agentic AI project.
"""

import logging
from pathlib import Path

from config import LOGS_DIR


class ProjectLogger:

    def __init__(self):

        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.system = self._create_logger(
            "system_logger",
            "system.log"
        )

        self.access = self._create_logger(
            "access_logger",
            "access.log"
        )

        self.audit = self._create_logger(
            "audit_logger",
            "audit.log"
        )

        self.security = self._create_logger(
            "security_logger",
            "security.log"
        )

        self.error = self._create_logger(
            "error_logger",
            "error.log"
        )

    def _create_logger(self, logger_name, filename):

        logger = logging.getLogger(logger_name)

        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            Path(LOGS_DIR) / filename,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    # ---------------------------------------------------
    # System
    # ---------------------------------------------------

    def log_system(self, message):

        self.system.info(message)

    # ---------------------------------------------------
    # Access
    # ---------------------------------------------------

    def log_access(self, username, status):

        self.access.info(
            f"User={username} | Status={status}"
        )

    # ---------------------------------------------------
    # Audit
    # ---------------------------------------------------

    def log_audit(self, username, action):

        self.audit.info(
            f"User={username} | Action={action}"
        )

    # ---------------------------------------------------
    # Security
    # ---------------------------------------------------

    def log_security(self, username, event):

        self.security.warning(
            f"User={username} | Event={event}"
        )

    # ---------------------------------------------------
    # Error
    # ---------------------------------------------------

    def log_error(self, error):

        self.error.error(str(error))


logger = ProjectLogger()


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Logger Test")
    print("=" * 60)

    logger.log_system("Application Started")

    logger.log_access(
        "s041",
        "SUCCESS"
    )

    logger.log_audit(
        "s041",
        "LOGIN"
    )

    logger.log_security(
        "s041",
        "Suspicious Behaviour"
    )

    logger.log_error(
        "Database Connection Failed"
    )

    print("All log files created successfully.")