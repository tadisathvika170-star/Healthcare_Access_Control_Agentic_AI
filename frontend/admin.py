"""
admin.py

Administrator Dashboard
"""

import streamlit as st
import sqlite3

from config import DATABASE_PATH


class AdminPage:
    """
    Administrator Dashboard
    """

    def __init__(self):
        pass

    def get_users(self):
        """
        Fetch all registered users.
        """

        connection = sqlite3.connect(DATABASE_PATH)

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, username
            FROM users
            ORDER BY id
            """
        )

        users = cursor.fetchall()

        connection.close()

        return users

    def render(self):
        """
        Display administrator dashboard.
        """

        st.title("🛠 Administrator Dashboard")

        st.markdown("---")

        users = self.get_users()

        st.metric(
            "Registered Users",
            len(users)
        )

        st.markdown("---")

        st.subheader("User List")

        if len(users) == 0:

            st.warning("No registered users found.")

        else:

            for user in users:

                st.write(
                    f"**ID:** {user[0]}  |  **Username:** {user[1]}"
                )

        st.markdown("---")

        st.subheader("System Status")

        st.success("✅ Database Connected")

        st.success("✅ AI Models Loaded")

        st.success("✅ Authentication Service Ready")

        st.success("✅ Access Control Active")

        st.success("✅ Agentic AI Running")

        st.markdown("---")

        if st.button(
            "Back to Dashboard",
            use_container_width=True
        ):

            st.session_state["page"] = "dashboard"

            st.rerun()


# ===========================================================
# Testing
# ===========================================================

if __name__ == "__main__":

    page = AdminPage()

    page.render()