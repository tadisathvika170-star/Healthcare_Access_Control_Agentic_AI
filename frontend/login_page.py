"""
login_page.py

Streamlit Login Page
"""

import streamlit as st


class LoginPage:
    """
    Login Page UI
    """

    def __init__(self):
        pass

    def render(self):
        """
        Display login form.

        Returns
        -------
        tuple
            (username, password, login_clicked)
        """

        st.title("🏥 Healthcare Access Control")
        st.subheader("Agentic AI Authentication System")

        st.write(
            "Please login using your credentials."
        )

        username = st.text_input(
            "Username",
            placeholder="Enter Username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter Password"
        )

        login_clicked = st.button(
            "Login",
            use_container_width=True
        )

        return (
            username,
            password,
            login_clicked
        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    page = LoginPage()
    page.render()