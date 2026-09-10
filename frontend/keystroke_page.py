"""Keystroke authentication page using one biometric verification repetition."""

import joblib
import numpy as np
import streamlit as st

from keystroke.component import KeystrokeComponent
from keystroke.feature_extractor import FeatureExtractor
from predict.predictor import Predictor
from database.patient_service import PatientService

SCALER_PATH = "models/scaler.pkl"
SEQUENCE_LENGTH = 1
INPUT_SIZE = 31
CONFIDENCE_THRESHOLD = 0.80


class KeystrokePage:
    def __init__(self):
        self.component = KeystrokeComponent()
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor()
        self.patient_service = PatientService()
        self.scaler = joblib.load(SCALER_PATH)

    def split_repetitions(self, events):
        repetitions = []
        current = []
        for event in events:
            current.append(event)
            if event.get("key") == "Enter":
                repetitions.append(current)
                current = []
        if current:
            repetitions.append(current)
        return repetitions

    def build_sequence(self, events):
        repetitions = self.split_repetitions(events)
        if len(repetitions) < SEQUENCE_LENGTH:
            raise ValueError("Not enough repetitions. Exactly 1 repetition is required.")

        repetition = repetitions[0]
        features = self.feature_extractor.extract(repetition)
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected 31 features, received {features.shape}.")

        raw_sequence = features.reshape(1, INPUT_SIZE)
        scaled_sequence = self.scaler.transform(raw_sequence).astype(np.float32)
        return scaled_sequence

    def process_submission(self, component_result, username, verification_mode):
        if not isinstance(component_result, dict) or not component_result.get("submitted", False):
            return

        submission_id = component_result.get("submission_id")
        last_submission = st.session_state.get("last_keystroke_submission_id")
        if submission_id is not None and submission_id == last_submission:
            return
        st.session_state["last_keystroke_submission_id"] = submission_id

        events = component_result.get("events", [])
        if not events:
            st.error("No keystroke data was received.")
            return

        try:
            sequence = self.build_sequence(events)
            result = self.predictor.predict(sequence)
        except Exception as error:
            st.error(f"Keystroke processing failed: {error}")
            return

        predicted_user = str(result.get("predicted_user", "Unknown"))
        confidence = float(result.get("confidence", 0.0))
        agreement = bool(result.get("agreement", False))

        st.session_state["keystroke_confidence"] = confidence
        st.session_state["keystroke_agreement"] = agreement
        st.session_state["keystroke_predicted_user"] = predicted_user

        st.success("Keystroke analysis completed.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted User", predicted_user)
        with col2:
            st.metric("Fusion Confidence", f"{confidence:.2%}")
        with col3:
            st.metric("Model Agreement", "Yes" if agreement else "No")

        st.write(f"GRU confidence: {float(result.get('gru_confidence', 0.0)):.2%}")
        st.write(f"LSTM confidence: {float(result.get('lstm_confidence', 0.0)):.2%}")

        identity_verified = predicted_user.lower() == str(username).lower()

        if not identity_verified:
            st.error("Keystroke pattern does not match the logged-in user.")
            if verification_mode == "patient_access":
                st.error("🔒 Patient details remain locked.")
            return

        if confidence < CONFIDENCE_THRESHOLD:
            st.error(f"Keystroke confidence is below the required {CONFIDENCE_THRESHOLD:.0%}.")
            if verification_mode == "patient_access":
                st.error("🔒 Patient details remain locked.")
            return

        if verification_mode == "patient_access":
            patient_id = st.session_state.get("pending_patient_id")
            doctor_username = st.session_state.get("pending_patient_doctor")
            if not patient_id:
                st.error("No patient access request is currently pending.")
                return
            if doctor_username and str(username).lower() != str(doctor_username).lower():
                st.error("You are not the doctor assigned to this patient.")
                st.error("🔒 Patient details remain locked.")
                return
            try:
                patient = self.patient_service.get_patient_by_id(patient_id)
            except Exception:
                patient = None
            if patient is None:
                st.error("Patient record could not be found.")
                return
            st.session_state["patient_access_verified"] = True
            st.success("Identity verified successfully.")
            st.success("🔓 Patient access verification passed.")
            return

        st.success("Keystroke identity verified successfully.")

    def render(self, username):
        st.title("⌨️ Keystroke Authentication")

        verification_mode = st.session_state.get("keystroke_verification_mode", "login")

        if verification_mode == "patient_access":
            doctor_username = st.session_state.get("pending_patient_doctor", "Unknown")
            st.warning(f"Assigned Doctor: {doctor_username}")

        st.markdown("---")
        st.subheader("⌨️ Behavioral Biometric Verification")
        st.write("Type the following phrase exactly 1 time:")
        st.code(".tie5Roanl")
        st.write("Press Enter after the repetition.")

        component_result = self.component.render(key="healthcare_keystroke_capture")
        if component_result:
            self.process_submission(component_result, username, verification_mode)
