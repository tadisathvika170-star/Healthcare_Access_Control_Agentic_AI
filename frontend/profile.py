"""
profile.py

User Profile Page
"""

import streamlit as st


class ProfilePage:
    """
    Displays the logged-in user's profile.
    """

    def __init__(self):
        pass

    def render(
        self,
        username,
        confidence=None,
        predicted_user=None,
        agreement=None
    ):
        """
        Render the profile page.
        """

        st.title("👤 User Profile")

        st.markdown("---")

        st.subheader("Basic Information")

        st.write(f"**Username:** {username}")

        if predicted_user is not None:
            st.write(f"**Predicted User:** {predicted_user}")

        if confidence is not None:
            st.write(f"**Authentication Confidence:** {confidence:.2%}")

        if agreement is not None:
            if agreement:
                st.success("LSTM and GRU predictions matched.")
            else:
                st.warning("LSTM and GRU predictions did not match.")

        st.markdown("---")

        st.subheader("Account Status")

        st.success("✅ Account Active")

        st.markdown("---")

        st.subheader("Authentication Methods")

        st.write("✔ Username & Password")
        st.write("✔ Keystroke Dynamics")
        st.write("✔ LSTM Verification")
        st.write("✔ GRU Verification")
        st.write("✔ Fusion Prediction")
        st.write("✔ Agentic AI Decision")

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

    profile = ProfilePage()

    profile.render(
        username="s041",
        confidence=0.991,
        predicted_user="s041",
        agreement=True
    )