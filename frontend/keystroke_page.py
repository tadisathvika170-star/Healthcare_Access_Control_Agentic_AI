"""
keystroke_page.py

Keystroke Authentication Page

Supports:
    1. Normal authentication
    2. Patient access verification

Patient access flow:

    Dashboard
        ↓
    View Details
        ↓
    Keystroke Verification
        ↓
    Verify User
        ↓
    Check Confidence
        ↓
    Check Doctor/Patient
        ↓
    Unlock Patient Details
"""

import streamlit as st
import numpy as np

from keystroke.component import KeystrokeComponent
from predict.predictor import Predictor
from database.patient_service import PatientService


class KeystrokePage:

    def __init__(self):

        self.component = KeystrokeComponent()

        self.predictor = Predictor()

        self.patient_service = PatientService()

    # ======================================================
    # GET PREDICTED USER SAFELY
    # ======================================================

    def get_predicted_user(self, result):

        """
        Different versions of Predictor may return
        different key names.

        This function prevents:

            KeyError: 'user'
        """

        if not isinstance(result, dict):

            return "Unknown"

        # First try the original key
        if result.get("user") is not None:

            return result.get("user")

        # Alternative key names
        if result.get("predicted_user") is not None:

            return result.get("predicted_user")

        if result.get("username") is not None:

            return result.get("username")

        if result.get("prediction") is not None:

            return result.get("prediction")

        return "Unknown"

    # ======================================================
    # GET CONFIDENCE SAFELY
    # ======================================================

    def get_confidence(self, result):

        if not isinstance(result, dict):

            return 0.0

        confidence = result.get(
            "confidence",
            0.0
        )

        try:

            return float(confidence)

        except:

            return 0.0

    # ======================================================
    # GET LSTM CONFIDENCE
    # ======================================================

    def get_lstm_confidence(self, result):

        if not isinstance(result, dict):

            return 0.0

        value = result.get(
            "lstm_confidence",
            0.0
        )

        try:

            return float(value)

        except:

            return 0.0

    # ======================================================
    # GET GRU CONFIDENCE
    # ======================================================

    def get_gru_confidence(self, result):

        if not isinstance(result, dict):

            return 0.0

        value = result.get(
            "gru_confidence",
            0.0
        )

        try:

            return float(value)

        except:

            return 0.0

    # ======================================================
    # GET MODEL AGREEMENT
    # ======================================================

    def get_agreement(self, result):

        if not isinstance(result, dict):

            return False

        return bool(
            result.get(
                "agreement",
                False
            )
        )

    # ======================================================
    # CREATE NOTIFICATION
    # ======================================================

    def create_notification(
        self,
        notification_type,
        username,
        predicted_user,
        confidence,
        agreement,
        patient=None
    ):

        notification_service = (
            st.session_state.get(
                "notification_service"
            )
        )

        if notification_service is None:

            return

        # ----------------------------------------------
        # Patient access success
        # ----------------------------------------------

        if notification_type == (
            "patient_access_success"
        ):

            notification = {

                "type":
                    "patient_access_success",

                "username":
                    username,

                "predicted_user":
                    predicted_user,

                "confidence":
                    confidence,

                "agreement":
                    agreement,

                "patient_id":
                    (
                        patient.get(
                            "patient_id"
                        )
                        if patient
                        else None
                    ),

                "patient_name":
                    (
                        patient.get(
                            "patient_name"
                        )
                        if patient
                        else None
                    ),

                "read":
                    False
            }

        # ----------------------------------------------
        # Patient access failed
        # ----------------------------------------------

        elif notification_type == (
            "patient_access_failed"
        ):

            notification = {

                "type":
                    "patient_access_failed",

                "username":
                    username,

                "predicted_user":
                    predicted_user,

                "confidence":
                    confidence,

                "agreement":
                    agreement,

                "patient_id":
                    (
                        patient.get(
                            "patient_id"
                        )
                        if patient
                        else None
                    ),

                "patient_name":
                    (
                        patient.get(
                            "patient_name"
                        )
                        if patient
                        else None
                    ),

                "read":
                    False
            }

        # ----------------------------------------------
        # Normal authentication check
        # ----------------------------------------------

        else:

            notification = {

                "type":
                    "authentication_check",

                "username":
                    username,

                "predicted_user":
                    predicted_user,

                "confidence":
                    confidence,

                "agreement":
                    agreement,

                "read":
                    False
            }

        # ----------------------------------------------
        # Add notification
        # ----------------------------------------------

        try:

            notification_service.add_notification(
                notification
            )

        except Exception:

            # If your NotificationService uses a
            # different method, don't crash the
            # authentication process.
            pass

    # ======================================================
    # RENDER
    # ======================================================

    def render(
        self,
        username
    ):

        st.title(
            "⌨️ Keystroke Authentication"
        )

        # ==================================================
        # DETERMINE VERIFICATION TYPE
        # ==================================================

        verification_mode = (
            st.session_state.get(
                "keystroke_verification_mode",
                "login"
            )
        )

        # ==================================================
        # PATIENT ACCESS MODE
        # ==================================================

        if verification_mode == (
            "patient_access"
        ):

            st.warning(
                "🔐 Patient Access Verification"
            )

            st.write(
                "Keystroke verification is required "
                "before viewing protected patient details."
            )

            patient_id = (
                st.session_state.get(
                    "pending_patient_id"
                )
            )

            patient = None

            # ----------------------------------------------
            # Find requested patient
            # ----------------------------------------------

            if patient_id:

                try:

                    # First try the selected patient
                    # saved by dashboard.

                    selected_patient = (
                        st.session_state.get(
                            "selected_patient"
                        )
                    )

                    if (
                        selected_patient
                        and
                        selected_patient.get(
                            "patient_id"
                        ) == patient_id
                    ):

                        patient = selected_patient

                except Exception:

                    patient = None

            # ----------------------------------------------
            # Display patient ID
            # ----------------------------------------------

            st.info(
                f"Patient ID requested: "
                f"**{patient_id}**"
            )

            if patient:

                st.write(
                    f"👤 Patient: "
                    f"**{patient.get('patient_name', 'Unknown')}**"
                )

                st.write(
                    f"👨‍⚕️ Assigned Doctor: "
                    f"**{patient.get('doctor_username', 'Unknown')}**"
                )

        # ==================================================
        # NORMAL LOGIN MODE
        # ==================================================

        else:

            st.info(
                "⌨️ Additional identity verification "
                "is required."
            )

        # ==================================================
        # TYPING INSTRUCTION
        # ==================================================

        st.markdown("---")

        st.subheader(
            "⌨️ Type Verification Text"
        )

        st.write(
            "Type the following phrase exactly "
            "using your normal typing pattern:"
        )

        st.code(
            ".tie5Roanl"
        )

        # ==================================================
        # KEYSTROKE COMPONENT
        # ==================================================

        self.component.render()

        st.markdown("---")

        # ==================================================
        # DEMO NOTICE
        # ==================================================

        st.warning(
            "Demo Mode: Prediction currently uses "
            "a sample feature vector. Connect the "
            "real browser keystroke features when "
            "your keystroke component is ready."
        )

        # ==================================================
        # VERIFY BUTTON
        # ==================================================

        if st.button(
            "⌨️ Verify Keystroke Pattern",
            use_container_width=True
        ):

            # ==================================================
            # TEMPORARY SAMPLE FEATURES
            # ==================================================

            sample_features = (
                np.random.rand(33)
            )

            # ==================================================
            # AI PREDICTION
            # ==================================================

            try:

                result = (
                    self.predictor.predict(
                        sample_features
                    )
                )

            except Exception as error:

                st.error(
                    "❌ Unable to perform "
                    "keystroke prediction."
                )

                st.exception(error)

                return

            # ==================================================
            # CHECK RESULT
            # ==================================================

            if not isinstance(
                result,
                dict
            ):

                st.error(
                    "❌ Predictor returned an "
                    "invalid result."
                )

                return

            # ==================================================
            # SAFELY READ RESULT
            # ==================================================

            predicted_user = (
                self.get_predicted_user(
                    result
                )
            )

            confidence = (
                self.get_confidence(
                    result
                )
            )

            lstm_confidence = (
                self.get_lstm_confidence(
                    result
                )
            )

            gru_confidence = (
                self.get_gru_confidence(
                    result
                )
            )

            agreement = (
                self.get_agreement(
                    result
                )
            )

            # ==================================================
            # DISPLAY RESULT
            # ==================================================

            st.subheader(
                "🧠 AI Prediction Result"
            )

            st.write(
                f"Expected User: "
                f"**{username}**"
            )

            st.write(
                f"Predicted User: "
                f"**{predicted_user}**"
            )

            st.write(
                f"Overall Confidence: "
                f"**{confidence:.4f}**"
            )

            st.write(
                f"LSTM Confidence: "
                f"**{lstm_confidence:.4f}**"
            )

            st.write(
                f"GRU Confidence: "
                f"**{gru_confidence:.4f}**"
            )

            # ==================================================
            # MODEL AGREEMENT
            # ==================================================

            if agreement:

                st.success(
                    "✅ LSTM and GRU models agree."
                )

            else:

                st.warning(
                    "⚠️ LSTM and GRU models disagree."
                )

            # ==================================================
            # UPDATE SESSION STATE
            # ==================================================

            st.session_state[
                "keystroke_confidence"
            ] = confidence

            st.session_state[
                "keystroke_agreement"
            ] = agreement

            st.session_state[
                "keystroke_predicted_user"
            ] = predicted_user

            # ==================================================
            # CHECK USER IDENTITY
            # ==================================================

            identity_verified = (
                predicted_user.lower()
                ==
                username.lower()
            )

            # ==================================================
            # PATIENT ACCESS VERIFICATION
            # ==================================================

            if verification_mode == (
                "patient_access"
            ):

                # ----------------------------------------------
                # Identity verification
                # ----------------------------------------------

                if not identity_verified:

                    self.create_notification(
                        notification_type=(
                            "patient_access_failed"
                        ),
                        username=username,
                        predicted_user=(
                            predicted_user
                        ),
                        confidence=confidence,
                        agreement=agreement
                    )

                    st.error(
                        "❌ Keystroke verification failed."
                    )

                    st.warning(
                        "⚠️ Please check whether "
                        "the user is authenticated or not."
                    )

                    st.error(
                        "🔒 Patient details remain locked."
                    )

                    return

                # ----------------------------------------------
                # Confidence verification
                # ----------------------------------------------

                if confidence < 0.75:

                    self.create_notification(
                        notification_type=(
                            "patient_access_failed"
                        ),
                        username=username,
                        predicted_user=(
                            predicted_user
                        ),
                        confidence=confidence,
                        agreement=agreement
                    )

                    st.error(
                        "❌ Keystroke confidence "
                        "is too low."
                    )

                    st.warning(
                        "⚠️ Please check whether "
                        "the user is authenticated "
                        "or not."
                    )

                    st.error(
                        "🔒 Patient details remain locked."
                    )

                    return

                # ----------------------------------------------
                # Get requested patient
                # ----------------------------------------------

                patient_id = (
                    st.session_state.get(
                        "pending_patient_id"
                    )
                )

                doctor_username = (
                    st.session_state.get(
                        "pending_patient_doctor"
                    )
                )

                # ----------------------------------------------
                # Validate required information
                # ----------------------------------------------

                if not patient_id:

                    st.error(
                        "❌ No patient was selected."
                    )

                    return

                if not doctor_username:

                    st.error(
                        "❌ Doctor information is missing."
                    )

                    return

                # ----------------------------------------------
                # Verify patient belongs to doctor
                # ----------------------------------------------

                try:

                    patient = (
                        self.patient_service
                        .get_patient(
                            patient_id,
                            doctor_username
                        )
                    )

                except Exception as error:

                    st.error(
                        "❌ Unable to verify "
                        "patient assignment."
                    )

                    st.exception(error)

                    return

                # ----------------------------------------------
                # Patient not assigned to doctor
                # ----------------------------------------------

                if patient is None:

                    self.create_notification(
                        notification_type=(
                            "patient_access_failed"
                        ),
                        username=username,
                        predicted_user=(
                            predicted_user
                        ),
                        confidence=confidence,
                        agreement=agreement
                    )

                    st.error(
                        "❌ Access denied."
                    )

                    st.error(
                        "This patient is not assigned "
                        "to the logged-in doctor."
                    )

                    st.error(
                        "🔒 Patient details remain locked."
                    )

                    return

                # ----------------------------------------------
                # SUCCESS
                # ----------------------------------------------

                self.create_notification(
                    notification_type=(
                        "patient_access_success"
                    ),
                    username=username,
                    predicted_user=(
                        predicted_user
                    ),
                    confidence=confidence,
                    agreement=agreement,
                    patient=patient
                )

                # ----------------------------------------------
                # Store verified patient
                # ----------------------------------------------

                st.session_state[
                    "verified_patient"
                ] = patient

                # ----------------------------------------------
                # Verification completed
                # ----------------------------------------------

                st.session_state[
                    "patient_verification_required"
                ] = False

                st.session_state[
                    "keystroke_verification_mode"
                ] = None

                # ----------------------------------------------
                # Clear pending information
                # ----------------------------------------------

                st.session_state[
                    "pending_patient_id"
                ] = None

                st.session_state[
                    "pending_patient_doctor"
                ] = None

                # ----------------------------------------------
                # Notification shown immediately
                # ----------------------------------------------

                st.success(
                    "🔔 Authentication Notification"
                )

                st.success(
                    "✅ Keystroke Authentication Successful"
                )

                st.write(
                    f"**User:** {username}"
                )

                st.write(
                    f"**Patient:** "
                    f"{patient['patient_name']}"
                )

                st.write(
                    f"**Patient ID:** "
                    f"{patient['patient_id']}"
                )

                st.write(
                    f"🎯 **Confidence:** "
                    f"{confidence * 100:.2f}%"
                )

                st.success(
                    "✅ User authenticated."
                )

                st.success(
                    "🔓 Patient details unlocked."
                )

                # ----------------------------------------------
                # Go back to dashboard
                # ----------------------------------------------

                st.session_state[
                    "page"
                ] = "Dashboard"

                st.rerun()

            # ==================================================
            # NORMAL LOGIN VERIFICATION
            # ==================================================

            else:

                # ----------------------------------------------
                # Identity matched
                # ----------------------------------------------

                if identity_verified:

                    # ------------------------------------------
                    # Confidence acceptable
                    # ------------------------------------------

                    if confidence >= 0.75:

                        st.success(
                            "✅ Identity Verified Successfully."
                        )

                        st.session_state.logged_in = True

                        st.session_state.username = (
                            username
                        )

                        st.session_state[
                            "keystroke_required"
                        ] = False

                        st.session_state[
                            "keystroke_verification_mode"
                        ] = None

                        st.success(
                            "🎉 Login Successful."
                        )

                        st.session_state[
                            "page"
                        ] = "Dashboard"

                        st.rerun()

                    # ------------------------------------------
                    # Confidence too low
                    # ------------------------------------------

                    else:

                        st.error(
                            "❌ Confidence is too low "
                            "for authentication."
                        )

                        st.warning(
                            "⚠️ Please check whether "
                            "the user is authenticated "
                            "or not."
                        )

                # ----------------------------------------------
                # Identity mismatch
                # ----------------------------------------------

                else:

                    self.create_notification(
                        notification_type=(
                            "authentication_check"
                        ),
                        username=username,
                        predicted_user=(
                            predicted_user
                        ),
                        confidence=confidence,
                        agreement=agreement
                    )

                    st.error(
                        "❌ Identity Verification Failed."
                    )

                    st.warning(
                        "⚠️ Please check whether "
                        "the user is authenticated "
                        "or not."
                    )

                    st.error(
                        "The keystroke pattern does not "
                        "match the logged-in user."
                    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    page = KeystrokePage()

    page.render(
        "testuser"
    )