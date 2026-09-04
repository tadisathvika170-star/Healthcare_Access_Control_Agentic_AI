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

from access_control import (
    authorize_access,
    USER_ROLES,
    ROLE_PERMISSIONS,
    CONFIDENCE_THRESHOLD
)
# ==========================================================
# STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Healthcare Access Control",
    page_icon="🏥",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if "username" not in st.session_state:

    st.session_state.username = ""


if "user_role" not in st.session_state:

    st.session_state.user_role = "Doctor"


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
# PATIENT VERIFICATION SESSION STATE
# ==========================================================

if "patient_verification_required" not in st.session_state:

    st.session_state[
        "patient_verification_required"
    ] = False


if "keystroke_verification_mode" not in st.session_state:

    st.session_state[
        "keystroke_verification_mode"
    ] = None


if "pending_patient_id" not in st.session_state:

    st.session_state[
        "pending_patient_id"
    ] = None


if "pending_patient_doctor" not in st.session_state:

    st.session_state[
        "pending_patient_doctor"
    ] = None


# ==========================================================
# NOTIFICATION SERVICE
# ==========================================================

if "notification_service" not in st.session_state:

    st.session_state[
        "notification_service"
    ] = NotificationService()


notification_service = (
    st.session_state[
        "notification_service"
    ]
)


# ==========================================================
# INITIALIZE PAGES
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
# LOGGED-IN SIDEBAR
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

    # Save selected page

    st.session_state[
        "page"
    ] = page

    # ======================================================
    # NOTIFICATION COUNT
    # ======================================================

    st.sidebar.markdown("---")

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

    # ======================================================
    # LOGOUT
    # ======================================================

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "🚪 Logout"
    ):

        # Clear authentication information

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.user_role = "Doctor"

        st.session_state.keystroke_required = False

        # Clear patient verification

        st.session_state[
            "patient_verification_required"
        ] = False

        st.session_state[
            "keystroke_verification_mode"
        ] = None

        st.session_state[
            "pending_patient_id"
        ] = None

        st.session_state[
            "pending_patient_doctor"
        ] = None

        st.session_state.pop(
            "verified_patient",
            None
        )

        st.rerun()


# ==========================================================
# NOT LOGGED IN
# ==========================================================

if not st.session_state.logged_in:

    # ======================================================
    # ADDITIONAL LOGIN KEYSTROKE
    # ======================================================

    if st.session_state.keystroke_required:

        st.warning(
            "⚠️ Additional authentication "
            "verification is required."
        )

        keystroke_page.render(
            st.session_state.username
        )

    # ======================================================
    # NORMAL LOGIN / REGISTER
    # ======================================================

    else:

        tab1, tab2 = st.tabs(

            [
                "Login",
                "Register"
            ]

        )

        # ==================================================
        # LOGIN
        # ==================================================

        with tab1:

            (
                username,
                password,
                clicked
            ) = login_page.render()

            if clicked:

                if not username or not password:

                    st.error(
                        "Please enter username "
                        "and password."
                    )

                else:

                    login_result = (
                        auth_service.login(
                            username,
                            password
                        )
                    )

                    if login_result:

                        # ----------------------------------
                        # LOGIN SUCCESS
                        # ----------------------------------

                        st.session_state.logged_in = True

                        st.session_state.username = (
                            username
                        )

                        # ----------------------------------
                        # Get role
                        # ----------------------------------

                        if isinstance(
                            login_result,
                            dict
                        ):

                            user_data = (
                                login_result.get(
                                    "user",
                                    {}
                                )
                            )

                            if isinstance(
                                user_data,
                                dict
                            ):

                                st.session_state[
                                    "user_role"
                                ] = user_data.get(
                                    "role",
                                    "Doctor"
                                )

                        st.session_state[
                            "page"
                        ] = "Dashboard"

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
        # REGISTER
        # ==================================================

        with tab2:

            register_page.render()


# ==========================================================
# LOGGED IN
# ==========================================================

else:

    username = (
        st.session_state.username
    )

    role = (
        st.session_state.get(
            "user_role",
            "Doctor"
        )
    )

    # ======================================================
    # PATIENT VERIFICATION HAS PRIORITY
    # ======================================================

    if (
        st.session_state.get(
            "patient_verification_required",
            False
        )
    ):

        keystroke_page.render(
            username
        )

    # ======================================================
    # DASHBOARD
    # ======================================================

    elif st.session_state.page == "Dashboard":

        dashboard.render(

            username=username,

            confidence=(
                st.session_state.get(
                    "keystroke_confidence"
                )
            ),

            agreement=(
                st.session_state.get(
                    "keystroke_agreement"
                )
            ),

            role=role

        )

    # ======================================================
    # PROFILE
    # ======================================================

    elif st.session_state.page == "Profile":

        profile.render(
            username
        )

    # ======================================================
    # KEYSTROKE
    # ======================================================

    elif st.session_state.page == "Keystroke":

        keystroke_page.render(
            username
        )

    # ======================================================
    # AUDIT LOGS
    # ======================================================

    elif st.session_state.page == "Audit Logs":

        audit_logs.render()

    # ======================================================
    # ADMIN
    # ======================================================

    elif st.session_state.page == "Admin":

        admin.render()

    # ======================================================
    # SETTINGS
    # ======================================================

    elif st.session_state.page == "Settings":

        settings.render()