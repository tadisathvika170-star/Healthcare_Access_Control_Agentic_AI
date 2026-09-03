"""
audit_agent.py

Professional Audit Agent

Stores all authentication events in audit_logs.db
for compliance, traceability, and security analysis.
"""

import sqlite3
from datetime import datetime


class AuditAgent:

    def __init__(self, db_path="database/audit_logs.db"):

        self.db_path = db_path

        self.initialize_database()

    # =====================================================
    # Initialize Database
    # =====================================================

    def initialize_database(self):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS audit_logs(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT,

            action TEXT,

            status TEXT,

            confidence REAL,

            risk_level TEXT,

            trust_score REAL,

            recommendation TEXT,

            timestamp TEXT

        )

        """)

        conn.commit()

        conn.close()

    # =====================================================
    # Log Authentication Event
    # =====================================================

    def log_event(

        self,

        username,

        action,

        status,

        confidence,

        risk_level,

        trust_score=0.0,

        recommendation=""

    ):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        cursor.execute("""

        INSERT INTO audit_logs(

            username,

            action,

            status,

            confidence,

            risk_level,

            trust_score,

            recommendation,

            timestamp

        )

        VALUES(?,?,?,?,?,?,?,?)

        """,

        (

            username,

            action,

            status,

            confidence,

            risk_level,

            trust_score,

            recommendation,

            datetime.now().strftime(

                "%Y-%m-%d %H:%M:%S"

            )

        )

        )

        conn.commit()

        conn.close()

    # =====================================================
    # Get All Logs
    # =====================================================

    def get_logs(self):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        cursor.execute("""

        SELECT *

        FROM audit_logs

        ORDER BY id DESC

        """)

        data = cursor.fetchall()

        conn.close()

        return data

    # =====================================================
    # Get User Logs
    # =====================================================

    def get_user_logs(self, username):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        cursor.execute("""

        SELECT *

        FROM audit_logs

        WHERE username=?

        ORDER BY id DESC

        """, (username,))

        data = cursor.fetchall()

        conn.close()

        return data

    # =====================================================
    # Authentication Statistics
    # =====================================================

    def statistics(self):

        logs = self.get_logs()

        total = len(logs)

        success = sum(

            1 for log in logs

            if log[3] == "SUCCESS"

        )

        failed = total - success

        return {

            "total_attempts": total,

            "successful": success,

            "failed": failed

        }


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    audit = AuditAgent()

    audit.log_event(

        username="doctor",

        action="LOGIN",

        status="SUCCESS",

        confidence=0.98,

        risk_level="LOW",

        trust_score=98.0,

        recommendation="Grant Access"

    )

    print(audit.statistics())