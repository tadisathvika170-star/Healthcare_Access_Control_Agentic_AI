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

st.set_page_config(
    page_title="Healthcare Access Control",
    page_icon="🏥",
    layout="wide"
)

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


def get_keystroke_page():
    if "_keystroke_page" not in st.session_state:
        from frontend.keystroke_page import KeystrokePage
        st.session_state["_keystroke_page"] = KeystrokePage()
    return st.session_state["_keystroke_page"]


def get_authentication_service():
    if "_authentication_service" not in st.session_state:
        from services.authentication_service import AuthenticationService
        st.session_state["_authentication_service"] = AuthenticationService()
    return st.session_state["_authentication_service"]


def get_notification_service():
    if "_notification_service" not in st.session_state:
        from services.notification_service import NotificationService
        st.session_state["_notification_service"] = NotificationService()
    return st.session_state["_notification_service"]


def render_sidebar():
    if not st.session_state.logged_in:
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select",
        ["Dashboard", "Profile", "Keystroke", "Audit Logs", "Admin", "Settings"]
    )
    st.session_state.page = page

    notification_service = get_notification_service()
    st.sidebar.markdown("---")

    try:
        unread = notification_service.get_unread_notifications()
    except Exception:
        unread = []

    if unread:
        st.sidebar.warning(f"🔔 {len(unread)} authentication notification(s)")
    else:
        st.sidebar.success("🔔 No new notifications")

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
        st.session_state.pop("verified_patient", None)

        for key in [
            "_login_page", "_register_page", "_dashboard", "_profile",
            "_admin", "_audit_logs", "_settings", "_keystroke_page",
            "_authentication_service", "_notification_service",
        ]:
            st.session_state.pop(key, None)

        st.session_state.page = "login"
        st.rerun()


def render_login():
    login_page = get_login_page()
    register_page = get_register_page()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username, password, clicked = login_page.render()

        if clicked:
            # Create the authentication service ONLY after Login is clicked.
            # This prevents the LSTM/GRU predictor from loading on the
            # Login/Register screen and makes the initial page load fast.
            auth_service = get_authentication_service()

            if not username or not password:
                st.error("Please enter username and password.")
            else:
                login_result = auth_service.login(username, password)

                if login_result:
                    st.session_state.logged_in = True
                    st.session_state.username = username

                    if isinstance(login_result, dict):
                        user_data = login_result.get("user", {})
                        if isinstance(user_data, dict):
                            st.session_state["user_role"] = user_data.get(
                                "role", "Doctor"
                            )

                    st.session_state.page = "Dashboard"
                    st.success("✅ Login Successful")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")

    with tab2:
        register_page.render()


def render_keystroke(username):
    keystroke_page = get_keystroke_page()
    keystroke_page.render(username)


def render_logged_in():
    username = st.session_state.username
    role = st.session_state.get("user_role", "Doctor")

    if st.session_state.get("patient_verification_required", False):
        render_keystroke(username)
        return

    if st.session_state.page == "Dashboard":
        dashboard = get_dashboard()
        dashboard.render(
            username=username,
            confidence=st.session_state.get("keystroke_confidence"),
            agreement=st.session_state.get("keystroke_agreement"),
            role=role
        )
        return

    if st.session_state.page == "Profile":
        get_profile().render(username)
        return

    if st.session_state.page == "Keystroke":
        render_keystroke(username)
        return

    if st.session_state.page == "Audit Logs":
        get_audit_logs().render()
        return

    if st.session_state.page == "Admin":
        get_admin().render()
        return

    if st.session_state.page == "Settings":
        get_settings().render()
        return


def main():
    render_sidebar()

    if not st.session_state.logged_in:
        if st.session_state.keystroke_required:
            st.warning("⚠️ Additional authentication verification is required.")
            render_keystroke(st.session_state.username)
        else:
            render_login()
        return

    render_logged_in()


if __name__ == "__main__":
    main()
