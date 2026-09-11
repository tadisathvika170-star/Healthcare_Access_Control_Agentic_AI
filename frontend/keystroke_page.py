"""Keystroke authentication page with fusion and Agentic AI results."""

import joblib
import numpy as np
import streamlit as st

from keystroke.component import KeystrokeComponent
from keystroke.feature_extractor import FeatureExtractor
from predict.predictor import Predictor
from database.patient_service import PatientService
from agents.confidence_agent import ConfidenceAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent

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
        self.confidence_agent = ConfidenceAgent()
        self.risk_agent = RiskAgent()
        self.strategy_agent = StrategyAgent()

    def split_repetitions(self, events):
        repetitions = []
        current = []
        for event in events:
            current.append(event)
            if event.get("key") in ("Enter", "Return"):
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

    def _inject_styles(self):
        st.markdown(
            """
            <style>
            .hac-banner {padding:16px 20px;border:1px solid #b7ead7;border-radius:10px;background:#ecfaf5;margin:8px 0 16px 0;}
            .hac-banner-title {font-size:18px;font-weight:700;color:#087f5b;margin-bottom:4px;}
            .hac-banner-text {font-size:13px;color:#31866d;}
            .hac-card {padding:18px 18px 14px 18px;border:1px solid #dbe7f5;border-radius:10px;background:#ffffff;min-height:125px;box-shadow:0 1px 5px rgba(20,50,90,.06);}
            .hac-card-purple {border-color:#ddd4ff;background:#fbf9ff;}
            .hac-card-green {border-color:#bdebdc;background:#f5fffb;}
            .hac-card-title {font-size:13px;color:#24456f;font-weight:600;margin-bottom:7px;}
            .hac-card-value {font-size:25px;color:#173d73;font-weight:700;margin-bottom:4px;}
            .hac-card-small {font-size:12px;color:#71839c;line-height:1.55;}
            .hac-section {font-size:19px;font-weight:700;color:#183b68;margin:18px 0 8px 0;}
            .hac-sub {font-size:12px;color:#71839c;margin-bottom:12px;}
            .agent-card {padding:12px 8px;text-align:center;border:1px solid #d8efe7;border-radius:8px;background:#f8fffc;min-height:110px;}
            .agent-icon {font-size:22px;margin-bottom:5px;}
            .agent-name {font-size:12px;font-weight:700;color:#24506d;}
            .agent-role {font-size:10px;color:#8295a8;min-height:28px;margin:3px 0 7px 0;}
            .agent-status {display:inline-block;padding:3px 9px;border-radius:12px;background:#d9f7eb;color:#13835f;font-size:10px;font-weight:700;}
            .fusion-panel {padding:16px;border:1px solid #dbe7f5;border-radius:10px;background:#fbfdff;}
            .fusion-row {display:flex;justify-content:space-between;gap:12px;padding:11px;border-radius:8px;background:#f4f8fd;margin-bottom:8px;font-size:12px;color:#526a85;}
            .fusion-result {padding:12px;border-left:4px solid #24a87b;background:#effbf6;border-radius:6px;font-size:13px;color:#176b53;}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _agent_card(self, icon, name, role, status="Completed"):
        return f"""
        <div class='agent-card'>
            <div class='agent-icon'>{icon}</div>
            <div class='agent-name'>{name}</div>
            <div class='agent-role'>{role}</div>
            <span class='agent-status'>{status}</span>
        </div>
        """

    def _render_agentic_dashboard(self, result, username, identity_verified):
        confidence = float(result.get("confidence", 0.0))
        agreement = bool(result.get("agreement", False))
        lstm_conf = float(result.get("lstm_confidence", 0.0))
        gru_conf = float(result.get("gru_confidence", 0.0))
        predicted_user = str(result.get("predicted_user", "Unknown"))

        confidence_result = self.confidence_agent.calculate(confidence)
        risk_result = self.risk_agent.evaluate(
            confidence=confidence,
            model_agreement=agreement,
            failed_attempts=0,
            trusted_user=identity_verified,
        )
        strategy_result = self.strategy_agent.choose_strategy(risk_result)

        access_granted = (
            identity_verified
            and confidence >= CONFIDENCE_THRESHOLD
            and agreement
        )
        decision_text = "Access Granted" if access_granted else "Access Denied"
        risk_level = risk_result.get("risk_level", "UNKNOWN")

        st.markdown(
            f"""
            <div class='hac-banner'>
                <div class='hac-banner-title'>✓ Keystroke verification completed</div>
                <div class='hac-banner-text'>User identity analysis is complete. The LSTM + GRU fusion result is now evaluated by the Agentic AI decision pipeline.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div class='hac-card'><div class='hac-card-title'>👤 Predicted User</div><div class='hac-card-value'>{predicted_user}</div><div class='hac-card-small'>GRU: {gru_conf:.2%}<br>LSTM: {lstm_conf:.2%}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='hac-card hac-card-purple'><div class='hac-card-title'>🛡 Fusion Confidence</div><div class='hac-card-value'>{confidence:.2%}</div><div class='hac-card-small'>Combined LSTM + GRU probability</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='hac-card'><div class='hac-card-title'>🔗 Fusion Mechanism</div><div class='hac-card-value' style='font-size:20px;'>Weighted Ensemble</div><div class='hac-card-small'>LSTM 50% + GRU 50%<br>Agreement: {'Yes' if agreement else 'No'}</div></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div class='hac-card hac-card-green'><div class='hac-card-title'>🤖 Agentic AI Decision</div><div class='hac-card-value' style='font-size:20px;'>{decision_text}</div><div class='hac-card-small'>Risk level: {risk_level}<br>Confidence level: {confidence_result.get('confidence_level', 'UNKNOWN')}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='hac-section'>🤖 Agentic AI Agents</div><div class='hac-sub'>Multi-agent analysis, risk evaluation, adaptive strategy and final access decision</div>", unsafe_allow_html=True)

        agents = [
            ("👤", "Identity Agent", "Biometric identity verification"),
            ("🔑", "Credential Agent", "Credential validation"),
            ("🛡", "Risk Agent", "Risk assessment"),
            ("📊", "Confidence Agent", "Model confidence"),
            ("✓", "Decision Agent", "Final access decision"),
            ("📄", "Audit Agent", "Authentication audit"),
            ("🧠", "Memory Agent", "Session memory"),
            ("⚙", "Strategy Agent", "Adaptive access strategy"),
        ]
        agent_cols = st.columns(8)
        for col, (icon, name, role) in zip(agent_cols, agents):
            with col:
                st.markdown(self._agent_card(icon, name, role), unsafe_allow_html=True)

        st.markdown("<div class='hac-section'>🔗 Fusion Mechanism Details</div>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"""
                <div class='fusion-panel'>
                    <div class='hac-card-title'>LSTM Model</div>
                    <div class='fusion-row'><span>Confidence</span><strong>{lstm_conf:.2%}</strong></div>
                    <div class='fusion-row'><span>Prediction</span><strong>{result.get('lstm_prediction', 'Unknown')}</strong></div>
                    <div class='fusion-row'><span>Sequence</span><strong>1 × 31</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class='fusion-panel'>
                    <div class='hac-card-title'>Feature-Enhanced GRU Model</div>
                    <div class='fusion-row'><span>Confidence</span><strong>{gru_conf:.2%}</strong></div>
                    <div class='fusion-row'><span>Prediction</span><strong>{result.get('gru_prediction', 'Unknown')}</strong></div>
                    <div class='fusion-row'><span>Agreement</span><strong>{'Yes' if agreement else 'No'}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='fusion-result'><strong>Fusion Result:</strong> Final prediction {predicted_user} &nbsp;•&nbsp; Confidence {confidence:.2%} &nbsp;•&nbsp; LSTM + GRU averaged probability distribution</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Agentic AI reasoning", expanded=False):
            st.write(f"Identity verified: {'Yes' if identity_verified else 'No'}")
            st.write(f"Confidence Agent: {confidence_result.get('confidence_level', 'UNKNOWN')}")
            st.write(f"Risk Agent: {risk_level} (score {risk_result.get('risk_score', 0)})")
            st.write(f"Strategy Agent: {strategy_result.get('action', 'UNKNOWN')}")
            for reason in risk_result.get("reasons", []):
                st.write(f"• {reason}")

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
        identity_verified = predicted_user.lower() == str(username).lower()

        st.session_state["keystroke_confidence"] = confidence
        st.session_state["keystroke_agreement"] = agreement
        st.session_state["keystroke_predicted_user"] = predicted_user

        st.success("Keystroke analysis completed.")
        self._render_agentic_dashboard(result, username, identity_verified)

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
        self._inject_styles()
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

        component_result = self.component.render(key="healthcare_keystroke_capture_v2")
        if component_result:
            self.process_submission(component_result, username, verification_mode)
