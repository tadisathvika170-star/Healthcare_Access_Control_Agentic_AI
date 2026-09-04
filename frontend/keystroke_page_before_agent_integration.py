"""
keystroke_page.py

Keystroke Authentication Page.

Supports:
    1. Normal authentication
    2. Patient access verification

Live pipeline:

    Browser keystrokes
        ↓
    10 password repetitions
        ↓
    31 features per repetition
        ↓
    StandardScaler
        ↓
    10 x 31 sequence
        ↓
    GRU + LSTM fusion
        ↓
    User prediction
        ↓
    Confidence / agreement verification
"""

import joblib
import numpy as np
import streamlit as st

from keystroke.component import KeystrokeComponent
from keystroke.feature_extractor import FeatureExtractor
from predict.predictor import Predictor
from database.patient_service import PatientService


# ============================================================
# Configuration
# ============================================================

SCALER_PATH = "models/scaler.pkl"

SEQUENCE_LENGTH = 10

INPUT_SIZE = 31

CONFIDENCE_THRESHOLD = 0.80


class KeystrokePage:

    def __init__(self):

        self.component = (
            KeystrokeComponent()
        )

        self.feature_extractor = (
            FeatureExtractor()
        )

        self.predictor = (
            Predictor()
        )

        self.patient_service = (
            PatientService()
        )

        self.scaler = joblib.load(
            SCALER_PATH
        )

    # ======================================================
    # GET PREDICTED USER
    # ======================================================

    def get_predicted_user(self, result):

        if not isinstance(result, dict):
            return "Unknown"

        value = result.get(
            "predicted_user"
        )

        if value is None:
            return "Unknown"

        return str(value)

    # ======================================================
    # GET CONFIDENCE
    # ======================================================

    def get_confidence(self, result):

        if not isinstance(result, dict):
            return 0.0

        try:

            return float(
                result.get(
                    "confidence",
                    0.0
                )
            )

        except Exception:

            return 0.0

    # ======================================================
    # GET LSTM CONFIDENCE
    # ======================================================

    def get_lstm_confidence(self, result):

        if not isinstance(result, dict):
            return 0.0

        try:

            return float(
                result.get(
                    "lstm_confidence",
                    0.0
                )
            )

        except Exception:

            return 0.0

    # ======================================================
    # GET GRU CONFIDENCE
    # ======================================================

    def get_gru_confidence(self, result):

        if not isinstance(result, dict):
            return 0.0

        try:

            return float(
                result.get(
                    "gru_confidence",
                    0.0
                )
            )

        except Exception:

            return 0.0

    # ======================================================
    # GET AGREEMENT
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

        notification = {

            "type":
                notification_type,

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

        if patient is not None:

            notification[
                "patient_id"
            ] = patient.get(
                "patient_id"
            )

            notification[
                "patient_name"
            ] = patient.get(
                "patient_name"
            )

        try:

            notification_service.add_notification(
                notification
            )

        except Exception:

            pass

    # ======================================================
    # SPLIT EVENTS INTO 3 REPETITIONS
    # ======================================================

    def split_repetitions(self, events):

        """
        Split the complete browser event stream
        at Enter key releases.

        Each repetition must end with Enter.
        """

        repetitions = []

        current = []

        for event in events:

            current.append(event)

            if event.get("key") == "Enter":

                repetitions.append(
                    current
                )

                current = []

        if current:

            repetitions.append(
                current
            )

        return repetitions

    # ======================================================
    # EXTRACT 10 x 31
    # ======================================================

    def build_sequence(self, events):

        repetitions = (
            self.split_repetitions(
                events
            )
        )

        if len(repetitions) < SEQUENCE_LENGTH:

            raise ValueError(
                "Not enough repetitions. "
                f"Expected {SEQUENCE_LENGTH}, "
                f"got {len(repetitions)}."
            )

        # Use exactly the first 3 complete repetitions.
        repetitions = repetitions[
            :SEQUENCE_LENGTH
        ]

        raw_features = []

        for index, repetition in enumerate(
            repetitions,
            start=1
        ):

            try:

                features = (
                    self.feature_extractor.extract(
                        repetition
                    )
                )

            except Exception as error:

                raise ValueError(
                    f"Repetition {index} could not "
                    f"be converted into 31 features: "
                    f"{error}"
                )

            if features.shape != (
                INPUT_SIZE,
            ):

                raise ValueError(
                    f"Repetition {index} produced "
                    f"{features.shape} instead of (31,)."
                )

            raw_features.append(
                features
            )

        raw_sequence = np.asarray(
            raw_features,
            dtype=np.float32
        )

        if raw_sequence.shape != (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ):

            raise ValueError(
                "Invalid sequence shape after "
                f"feature extraction: "
                f"{raw_sequence.shape}"
            )

        # --------------------------------------------------
        # Apply the SAME StandardScaler used during training
        # --------------------------------------------------

        scaled_sequence = self.scaler.transform(
            raw_sequence
        )

        scaled_sequence = np.asarray(
            scaled_sequence,
            dtype=np.float32
        )

        if scaled_sequence.shape != (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ):

            raise ValueError(
                "Invalid scaled sequence shape: "
                f"{scaled_sequence.shape}"
            )

        return scaled_sequence

    # ======================================================
    # PROCESS PREDICTION
    # ======================================================

    def process_submission(
        self,
        component_result,
        username,
        verification_mode
    ):

        if not isinstance(
            component_result,
            dict
        ):

            return

        if not component_result.get(
            "submitted",
            False
        ):

            return

        submission_id = (
            component_result.get(
                "submission_id"
            )
        )

        # --------------------------------------------------
        # Prevent processing the same submission twice
        # --------------------------------------------------

        last_submission = (
            st.session_state.get(
                "last_keystroke_submission_id"
            )
        )

        if (
            submission_id is not None
            and
            submission_id == last_submission
        ):

            return

        st.session_state[
            "last_keystroke_submission_id"
        ] = submission_id

        events = (
            component_result.get(
                "events",
                []
            )
        )

        if not events:

            st.error(
                "No keystroke data was received."
            )

            return

        # --------------------------------------------------
        # Build 3 x 31 sequence
        # --------------------------------------------------

        try:

            sequence = (
                self.build_sequence(
                    events
                )
            )

        except Exception as error:

            st.error(
                f"Keystroke processing failed: {error}"
            )

            return

        # --------------------------------------------------
        # AI prediction
        # --------------------------------------------------

        try:

            result = (
                self.predictor.predict(
                    sequence
                )
            )

        except Exception as error:

            st.error(
                "AI prediction failed."
            )

            st.exception(error)

            return

        # --------------------------------------------------
        # Read prediction
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Store prediction
        # --------------------------------------------------

        st.session_state[
            "keystroke_confidence"
        ] = confidence

        st.session_state[
            "keystroke_agreement"
        ] = agreement

        st.session_state[
            "keystroke_predicted_user"
        ] = predicted_user

        # --------------------------------------------------
        # Display model results
        # --------------------------------------------------

        st.success(
            "Keystroke analysis completed."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted User",
                predicted_user
            )

        with col2:

            st.metric(
                "Fusion Confidence",
                f"{confidence:.2%}"
            )

        with col3:

            st.metric(
                "Model Agreement",
                "Yes"
                if agreement
                else "No"
            )

        st.write(
            f"GRU confidence: "
            f"{gru_confidence:.2%}"
        )

        st.write(
            f"LSTM confidence: "
            f"{lstm_confidence:.2%}"
        )

        # --------------------------------------------------
        # Identity verification
        # --------------------------------------------------

        identity_verified = (
            predicted_user.lower()
            ==
            str(username).lower()
        )

        # ==================================================
        # PATIENT ACCESS MODE
        # ==================================================

        if verification_mode == (
            "patient_access"
        ):

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

            # ------------------------------------------------
            # Patient must exist
            # ------------------------------------------------

            if not patient_id:

                st.error(
                    "No patient access request "
                    "is currently pending."
                )

                return

            # ------------------------------------------------
            # Identity
            # ------------------------------------------------

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
                    "Keystroke identity does not "
                    "match the logged-in user."
                )

                st.error(
                    "🔒 Patient details remain locked."
                )

                return

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            if confidence < (
                CONFIDENCE_THRESHOLD
            ):

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
                    "Keystroke confidence is below "
                    f"the required "
                    f"{CONFIDENCE_THRESHOLD:.0%}."
                )

                st.error(
                    "🔒 Patient details remain locked."
                )

                return

            # ------------------------------------------------
            # Doctor verification
            # ------------------------------------------------

            if (
                doctor_username
                and
                str(username).lower()
                !=
                str(doctor_username).lower()
            ):

                st.error(
                    "You are not the doctor assigned "
                    "to this patient."
                )

                st.error(
                    "🔒 Patient details remain locked."
                )

                return

            # ------------------------------------------------
            # Successful patient verification
            # ------------------------------------------------

            patient = None

            try:

                patient = (
                    self.patient_service
                    .get_patient_by_id(
                        patient_id
                    )
                )

            except Exception:

                patient = None

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

            st.success(
                "Identity verified successfully."
            )

            st.success(
                "🔓 Patient access verification passed."
            )

            st.session_state[
                "patient_access_verified"
            ] = True

            return

        # ==================================================
        # NORMAL LOGIN MODE
        # ==================================================

        if not identity_verified:

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
                "Keystroke pattern does not match "
                "the logged-in user."
            )

            return

        if confidence < (
            CONFIDENCE_THRESHOLD
        ):

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
                "Keystroke confidence is below "
                f"the required "
                f"{CONFIDENCE_THRESHOLD:.0%}."
            )

            return

        # --------------------------------------------------
        # Successful authentication
        # --------------------------------------------------

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

        st.success(
            "Keystroke identity verified successfully."
        )

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

            doctor_username = (
                st.session_state.get(
                    "pending_patient_doctor",
                    "Unknown"
                )
            )

            st.warning(
                "🏥 Assigned Doctor: "
                f"**{doctor_username}**"
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
        # INSTRUCTIONS
        # ==================================================

        st.markdown("---")

        st.subheader(
            "⌨️ Behavioral Biometric Verification"
        )

        st.write(
            "Type the following phrase exactly "
            "10 times:"
        )

        st.code(
            ".tie5Roanl"
        )

        st.write(
            "Press Enter after every repetition."
        )

        # ==================================================
        # COMPONENT
        # ==================================================

        component_result = (
            self.component.render()
        )

        # ==================================================
        # PROCESS RESULT
        # ==================================================

        self.process_submission(
            component_result,
            username,
            verification_mode
        )


