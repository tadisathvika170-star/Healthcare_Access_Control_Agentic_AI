"""
audit_logs.py

Displays authentication audit logs.
"""

import sqlite3
import pandas as pd
import streamlit as st

from config import AUDIT_DATABASE_PATH


class AuditLogsPage:
    """
    Displays all authentication logs.
    """

    def __init__(self):
        pass

    def load_logs(self):
        """
        Load audit logs from database.
        """

        try:

            connection = sqlite3.connect(
                AUDIT_DATABASE_PATH
            )

            query = """
            SELECT
                id,
                username,
                action,
                status,
                timestamp
            FROM audit_logs
            ORDER BY id DESC
            """

            dataframe = pd.read_sql_query(
                query,
                connection
            )

            connection.close()

            return dataframe

        except Exception:

            return pd.DataFrame(
                columns=[
                    "id",
                    "username",
                    "action",
                    "status",
                    "timestamp"
                ]
            )

    def render(self):
        """
        Display audit logs.
        """

        st.title("📋 Audit Logs")

        st.markdown("---")

        dataframe = self.load_logs()

        if dataframe.empty:

            st.info("No audit logs available.")

        else:

            st.dataframe(
                dataframe,
                use_container_width=True
            )

            st.markdown("---")

            st.metric(
                "Total Logs",
                len(dataframe)
            )

            success = len(
                dataframe[
                    dataframe["status"] == "SUCCESS"
                ]
            )

            failure = len(
                dataframe[
                    dataframe["status"] == "FAILED"
                ]
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Successful Logins",
                    success
                )

            with col2:

                st.metric(
                    "Failed Logins",
                    failure
                )

        st.markdown("---")

        if st.button(
            "Back to Dashboard",
            use_container_width=True
        ):

            st.session_state["page"] = "dashboard"

            st.rerun()


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    page = AuditLogsPage()

    page.render()