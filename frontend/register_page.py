"""
register_page.py

User Registration Page
"""

import streamlit as st
from authentication.auth_manager import AuthManager


class RegisterPage:

    def __init__(self):

        self.auth = AuthManager()

    def render(self):

        st.title("📝 User Registration")

        st.write("Create a new account.")

        username = st.text_input(
            "Username",
            key="reg_username"
        )

        full_name = st.text_input(
            "Full Name",
            key="reg_fullname"
        )

        role = st.selectbox(
            "Role",
            [
                "Doctor",
                "Nurse",
                "Patient",
                "Admin"
            ],
            key="reg_role"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="reg_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="reg_confirm"
        )

        if st.button(
            "Register",
            use_container_width=True
        ):

            if username.strip() == "":
                st.error("Username cannot be empty.")
                return

            if full_name.strip() == "":
                st.error("Full name cannot be empty.")
                return

            if password != confirm_password:
                st.error("Passwords do not match.")
                return

            if len(password) < 6:
                st.error("Password must contain at least 6 characters.")
                return

            success, message = self.auth.register_user(
                username,
                password,
                full_name,
                role
            )

            if success:
                st.success(message)
            else:
                st.error(message)


if __name__ == "__main__":

    RegisterPage().render()
