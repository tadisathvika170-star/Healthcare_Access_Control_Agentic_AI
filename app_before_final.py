"""
app.py

Main entry point for the Healthcare Access Control
Agentic AI System.

Optimized version:
- Heavy keystroke/AI page is loaded only when required.
- Page objects are created only when needed.
- Existing login, dashboard, profile, admin, audit logs,
  settings, and patient verification flows are preserved.
"""

import streamlit as st


# ==========================================================
# STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Healthcare Access Control",
    page_icon="🏥",
    layout="wide"
)


# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================

DEFAULT_STATE = {
    "logged_in": False,
    "username": "",
    "user_role": "Doctor",
    "page": "login",

    "keystroke_required": False,
    "keystroke_confidence": None,
    "keystroke_agreement": None,
    "keystroke_predicted_user": None,

    "patient_verification_required": False,
    "keystroke_verification_mode": None,
    "pending_patient_id": None,
    "pending_patient_doctor": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# LAZY SERVICE / PAGE LOADERS
# ==========================================================
#
# These functions prevent heavy modules from loading when
# the application is only showing the login screen.
# ==========================================================


def get_login_page():

    if "_login_page" not in st.session_state:

        from frontend.login_page import LoginPage

        st.session_state["_login_page"] = LoginPage()

    return st.session_state["_login_page"]


def get_register_page():

    if "_register_page" not in st.session_state:

        from frontend.register_page import RegisterPage

        st.session_state["_register_page"] = RegisterPage()

    return st.session_state["_register_page"]


def get_dashboard():

    if "_dashboard" not in st.session_state:

        from frontend.dashboard import Dashboard

        st.session_state["_dashboard"] = Dashboard()

    return st.session_state["_dashboard"]


def get_profile():

    if "_profile" not in st.session_state:

        from frontend.profile import ProfilePage

        st.session_state["_profile"] = ProfilePage()

    return st.session_state["_profile"]


def get_admin():

    if "_admin" not in st.session_state:

        from frontend.admin import AdminPage

        st.session_state["_admin"] = AdminPage()

    return st.session_state["_admin"]


def get_audit_logs():

    if "_audit_logs" not in st.session_state:

        from frontend.audit_logs import AuditLogsPage

        st.session_state["_audit_logs"] = AuditLogsPage()

    return st.session_state["_audit_logs"]


def get_settings():

    if "_settings" not in st.session_state:

        from frontend.settings import SettingsPage

        st.session_state["_settings"] = SettingsPage()

    return st.session_state["_settings"]


# ==========================================================
# IMPORTANT:
# KEYSTROKE PAGE IS HEAVY
#
# Do NOT import it at application startup.
#
# It will only load when the user actually reaches
# Keystroke Authentication.
# ==========================================================


def get_keystroke_page():

    if "_keystroke_page" not in st.session_state:

        from frontend.keystroke_page import KeystrokePage

        st.session_state["_keystroke_page"] = KeystrokePage()

    return st.session_state["_keystroke_page"]


# ==========================================================
# SERVICES
# ==========================================================


def get_authentication_service():

    if "_authentication_service" not in st.session_state:

        from services.authentication_service import (
            AuthenticationService
        )

        st.session_state[
            "_authentication_service"
        ] = AuthenticationService()

    return st.session_state[
        "_authentication_service"
    ]


def get_notification_service():

    if "_notification_service" not in st.session_state:

        from services.notification_service import (
            NotificationService
        )

        st.session_state[
            "_notification_service"
        ] = NotificationService()

    return st.session_state[
        "_notification_service"
    ]


# ==========================================================
# SIDEBAR
# ==========================================================


def render_sidebar():

    if not st.session_state.logged_in:
        return

    st.sidebar.title("Navigation")

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

    st.session_state.page = page

    # ------------------------------------------------------
    # Notifications
    # ------------------------------------------------------

    notification_service = get_notification_service()

    st.sidebar.markdown("---")

    try:

        unread = (
            notification_service
            .get_unread_notifications()
        )

    except Exception:

        unread = []

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

    if st.sidebar.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_role = "Doctor"

        st.session_state.keystroke_required = False

        st.session_state.keystroke_confidence = None
        st.session_state.keystroke_agreement = None
        st.session_state.keystroke_predicted_user = None

        st.session_state.patient_verification_required = False
        st.session_state.keystroke_verification_mode = None
        st.session_state.pending_patient_id = None
        st.session_state.pending_patient_doctor = None

        st.session_state.pop(
            "verified_patient",
            None
        )

        # Remove cached page objects so the next login
        # starts cleanly.
        for key in [
            "_login_page",
            "_register_page",
            "_dashboard",
            "_profile",
            "_admin",
            "_audit_logs",
            "_settings",
            "_keystroke_page",
            "_authentication_service",
            "_notification_service",
        ]:

            st.session_state.pop(
                key,
                None
            )

        st.session_state.page = "login"

        st.rerun()


# ==========================================================
# LOGIN / REGISTER PAGE
# ==========================================================


def render_login():

    login_page = get_login_page()
    register_page = get_register_page()

    auth_service = get_authentication_service()

    tab1, tab2 = st.tabs(
        [
            "Login",
            "Register"
        ]
    )

    # ======================================================
    # LOGIN
    # ======================================================

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

                    # --------------------------------------
                    # LOGIN SUCCESS
                    # --------------------------------------

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        username
                    )

                    # --------------------------------------
                    # Get role
                    # --------------------------------------

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

                    st.session_state.page = (
                        "Dashboard"
                    )

                    st.success(
                        "✅ Login Successful"
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Invalid Username "
                        "or Password"
                    )

    # ======================================================
    # REGISTER
    # ======================================================

    with tab2:

        register_page.render()


# ==========================================================
# KEYSTROKE VERIFICATION
# ==========================================================


def render_keystroke(username):

    keystroke_page = get_keystroke_page()

    keystroke_page.render(
        username
    )


# ==========================================================
# LOGGED-IN PAGE ROUTER
# ==========================================================


def render_logged_in():

    username = st.session_state.username

    role = st.session_state.get(
        "user_role",
        "Doctor"
    )

    # ======================================================
    # PATIENT VERIFICATION HAS PRIORITY
    # ======================================================

    if st.session_state.get(
        "patient_verification_required",
        False
    ):

        render_keystroke(
            username
        )

        return

    # ======================================================
    # DASHBOARD
    # ======================================================

    if st.session_state.page == "Dashboard":

        dashboard = get_dashboard()

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

        return

    # ======================================================
    # PROFILE
    # ======================================================

    if st.session_state.page == "Profile":

        profile = get_profile()

        profile.render(
            username
        )

        return

    # ======================================================
    # KEYSTROKE
    # ======================================================

    if st.session_state.page == "Keystroke":

        render_keystroke(
            username
        )

        return

    # ======================================================
    # AUDIT LOGS
    # ======================================================

    if st.session_state.page == "Audit Logs":

        audit_logs = get_audit_logs()

        audit_logs.render()

        return

    # ======================================================
    # ADMIN
    # ======================================================

    if st.session_state.page == "Admin":

        admin = get_admin()

        admin.render()

        return

    # ======================================================
    # SETTINGS
    # ======================================================

    if st.session_state.page == "Settings":

        settings = get_settings()

        settings.render()

        return


# ==========================================================
# MAIN APPLICATION
# ==========================================================


def main():

    # ------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------

    render_sidebar()

    # ======================================================
    # NOT LOGGED IN
    # ======================================================

    if not st.session_state.logged_in:

        # --------------------------------------------------
        # Additional login verification
        # --------------------------------------------------

        if st.session_state.keystroke_required:

            st.warning(
                "⚠️ Additional authentication "
                "verification is required."
            )

            render_keystroke(
                st.session_state.username
            )

        # --------------------------------------------------
        # Normal Login / Register
        # --------------------------------------------------

        else:

            render_login()

        return

    # ======================================================
    # LOGGED IN
    # ======================================================

    render_logged_in()


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================


if __name__ == "__main__":

    main()