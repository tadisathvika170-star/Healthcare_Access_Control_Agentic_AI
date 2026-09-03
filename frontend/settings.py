"""
settings.py

Application Settings Page
"""

import streamlit as st


class SettingsPage:
    """
    Application Settings Page
    """

    def __init__(self):
        pass

    def render(self):
        """
        Display application settings.
        """

        st.title("⚙️ Settings")

        st.markdown("---")

        st.subheader("Authentication")

        confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.01
        )

        st.write(
            f"Current Threshold : {confidence:.2f}"
        )

        st.markdown("---")

        st.subheader("Model Information")

        st.success("✔ LSTM Model Loaded")

        st.success("✔ GRU Model Loaded")

        st.success("✔ Fusion Prediction Enabled")

        st.success("✔ Agentic AI Enabled")

        st.markdown("---")

        st.subheader("Application")

        st.checkbox(
            "Enable Audit Logging",
            value=True
        )

        st.checkbox(
            "Enable Notifications",
            value=True
        )

        st.checkbox(
            "Enable Agent Decisions",
            value=True
        )

        st.markdown("---")

        if st.button(
            "Save Settings",
            use_container_width=True
        ):

            st.success(
                "Settings Saved Successfully."
            )

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

    page = SettingsPage()

    page.render()