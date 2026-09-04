"""
app.py

Main entry point for the Healthcare Access Control
Agentic AI System.
"""

import streamlit as st

from frontend.login_page import LoginPage
from frontend.register_page import RegisterPage
from frontend.dashboard import Dashboard
from frontend.profile import ProfilePage
from frontend.admin import AdminPage
from frontend.audit_logs import AuditLogsPage
from frontend.keystroke_page import KeystrokePage
from frontend.settings import SettingsPage

from services.authentication_service import (
    AuthenticationService
)

from services.notification_service import (
    NotificationService
)


# ==========================================================
# Streamlit Configuration
# ==========================================================

st.set_page_config(
    page_title="Healthcare Access Control",
    page_icon="🏥",
    layout="wide"
)


# ==========================================================
# Session State
# ==========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if "username" not in st.session_state:

    st.session_state.username = ""


if "page" not in st.session_state:

    st.session_state.page = "login"


if "keystroke_required" not in st.session_state:

    st.session_state.keystroke_required = False


if "keystroke_confidence" not in st.session_state:

    st.session_state.keystroke_confidence = None


if "keystroke_agreement" not in st.session_state:

    st.session_state.keystroke_agreement = None


if "keystroke_predicted_user" not in st.session_state:

    st.session_state.keystroke_predicted_user = None


# ==========================================================
# Notification Service
# ==========================================================

if "notification_service" not in st.session_state:

    st.session_state.notification_service = (
        NotificationService()
    )


notification_service = (
    st.session_state.notification_service
)


# ==========================================================
# Initialize Pages
# ==========================================================

login_page = LoginPage()

register_page = RegisterPage()

dashboard = Dashboard()

profile = ProfilePage()

admin = AdminPage()

audit_logs = AuditLogsPage()

keystroke_page = KeystrokePage()

settings = SettingsPage()

auth_service = AuthenticationService()


# ==========================================================
# Sidebar
# ==========================================================

if st.session_state.logged_in:

    st.sidebar.title(
        "Navigation"
    )

    page = st.sidebar.radio(

        "Select",

        [

            "Dashboard",

            "Profile",

            "Keystroke",

            "Audit Logs",

            "Admin",

            "Settings"

        ]

    )

    st.sidebar.markdown("---")

    # ------------------------------------------------------
    # Notification Count
    # ------------------------------------------------------

    unread = (
        notification_service
        .get_unread_notifications()
    )

    if unread:

        st.sidebar.warning(
            f"🔔 {len(unread)} "
            f"authentication notification(s)"
        )

    else:

        st.sidebar.success(
            "🔔 No new notifications"
        )

    # ------------------------------------------------------
    # Logout
    # ------------------------------------------------------

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "Logout"
    ):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.keystroke_required = False

        st.rerun()


# ==========================================================
# Login Screen
# ==========================================================

if not st.session_state.logged_in:

    # ------------------------------------------------------
    # Keystroke Verification Required
    # ------------------------------------------------------

    if st.session_state.keystroke_required:

        st.warning(
            "⚠️ Additional authentication "
            "verification is required."
        )

        st.info(
            "The system has requested keystroke "
            "pattern verification."
        )

        keystroke_page.render(
            st.session_state.username
        )

    # ------------------------------------------------------
    # Normal Login
    # ------------------------------------------------------

    else:

        tab1, tab2 = st.tabs(

            [

                "Login",

                "Register"

            ]

        )

        # ==================================================
        # LOGIN TAB
        # ==================================================

        with tab1:

            (
                username,
                password,
                clicked
            ) = login_page.render()

            if clicked:

                # ------------------------------------------
                # Basic Validation
                # ------------------------------------------

                if not username or not password:

                    st.error(
                        "Please enter username "
                        "and password."
                    )

                else:

                    # --------------------------------------
                    # Username + Password
                    # --------------------------------------

                    login_result = (
                        auth_service.login(
                            username,
                            password
                        )
                    )

                    if login_result:

                        # Normal login
                        st.session_state.logged_in = True

                        st.session_state.username = username

                        st.success(
                            "✅ Login Successful"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "❌ Invalid Username "
                            "or Password"
                        )

        # ==================================================
        # REGISTER TAB
        # ==================================================

        with tab2:

            register_page.render()


# ==========================================================
# Logged In
# ==========================================================

else:

    username = (
        st.session_state.username
    )

    # ------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------

    if page == "Dashboard":

        dashboard.render(

            username=username,

            confidence=(
                st.session_state
                .keystroke_confidence
            ),

            agreement=(
                st.session_state
                .keystroke_agreement
            )

        )

    # ------------------------------------------------------
    # Profile
    # ------------------------------------------------------

    elif page == "Profile":

        profile.render(
            username
        )

    # ------------------------------------------------------
    # Keystroke
    # ------------------------------------------------------

    elif page == "Keystroke":

        keystroke_page.render(
            username
        )

    # ------------------------------------------------------
    # Audit Logs
    # ------------------------------------------------------

    elif page == "Audit Logs":

        audit_logs.render()

    # ------------------------------------------------------
    # Admin
    # ------------------------------------------------------

    elif page == "Admin":

        admin.render()

    # ------------------------------------------------------
    # Settings
    # ------------------------------------------------------

    elif page == "Settings":

        settings.render()