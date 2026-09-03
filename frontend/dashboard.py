"""
frontend/dashboard.py

Healthcare Access Control Dashboard

Dashboard design follows the project's handwritten workflow:

    Login
       -> Patient / Doctor
       -> Patient Records
       -> View / Edit request
       -> Keystroke Dynamics
       -> LSTM + GRU
       -> Authentication
       -> Agentic AI
       -> Allow / Alert / Block

This file is intentionally limited to the dashboard UI and session-state
handoffs. It does not replace the trained LSTM/GRU models or the existing
keystroke verification page.
"""

import math

import streamlit as st

from database.patient_service import PatientService


class Dashboard:
    """Main healthcare access-control dashboard."""

    def __init__(self):
        self.patient_service = PatientService()

    # ==========================================================
    # MAIN DASHBOARD
    # ==========================================================

    def render(
        self,
        username,
        confidence=None,
        agreement=None,
        role="Doctor"
    ):
        """Render the complete healthcare dashboard."""

        # Always use the latest session values when available.
        confidence = st.session_state.get(
            "keystroke_confidence",
            confidence
        )

        agreement = st.session_state.get(
            "keystroke_agreement",
            agreement
        )

        predicted_user = st.session_state.get(
            "keystroke_predicted_user"
        )

        self._inject_styles()

        self._render_header(
            username,
            role,
            confidence,
            agreement,
            predicted_user
        )

        self._render_security_banner(
            username,
            confidence,
            agreement,
            predicted_user
        )

        self._render_quick_status(
            confidence,
            agreement,
            predicted_user
        )

        st.markdown("## 👨‍⚕️ Patient Records")

        st.caption(
            "Select a patient to request protected access. "
            "Patient details remain locked until keystroke "
            "verification succeeds."
        )

        patients = self._get_filtered_patients()

        self._render_patient_records(
            patients,
            username,
            role
        )

        self._render_verified_patient()

        self._render_ai_security_panel(
            confidence,
            agreement,
            predicted_user,
            username
        )

        self._render_notifications(
            username
        )

        self._render_system_status()

    # ==========================================================
    # STYLES
    # ==========================================================

    def _inject_styles(self):

        st.markdown(
            """
            <style>

            .main-title {
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.15rem;
            }

            .subtitle {
                font-size: 1rem;
                opacity: 0.78;
                margin-bottom: 1.2rem;
            }

            .status-card {
                border: 1px solid rgba(128,128,128,.25);
                border-radius: 14px;
                padding: 18px;
                min-height: 110px;
                background: rgba(128,128,128,.05);
            }

            .status-label {
                font-size: .82rem;
                opacity: .72;
                margin-bottom: 5px;
            }

            .status-value {
                font-size: 1.2rem;
                font-weight: 700;
            }

            .patient-card {
                border: 1px solid rgba(128,128,128,.25);
                border-radius: 16px;
                padding: 18px;
                margin: 10px 0;
                background: rgba(128,128,128,.035);
            }

            .patient-name {
                font-size: 1.15rem;
                font-weight: 750;
            }

            .patient-id {
                font-size: .82rem;
                opacity: .65;
            }

            .security-box {
                border: 1px solid rgba(128,128,128,.25);
                border-radius: 16px;
                padding: 18px;
                margin-top: 10px;
                background: rgba(128,128,128,.035);
            }

            </style>
            """,
            unsafe_allow_html=True
        )

    # ==========================================================
    # HEADER
    # ==========================================================

    def _render_header(
        self,
        username,
        role,
        confidence,
        agreement,
        predicted_user
    ):

        st.markdown(
            '<div class="main-title">'
            '🏥 Healthcare Access Control'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Secure patient-record access using credentials, '
            'keystroke dynamics, LSTM + GRU and Agentic AI.'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            self._status_card(
                "Logged-in User",
                username or "Unknown",
                "👤"
            )

        with col2:

            self._status_card(
                "Role",
                role or "Unknown",
                "🪪"
            )

        with col3:

            if confidence is None:

                value = "Not verified"

            else:

                value = (
                    f"{self._safe_confidence(confidence) * 100:.1f}%"
                )

            self._status_card(
                "Keystroke Confidence",
                value,
                "⌨️"
            )

        with col4:

            if predicted_user:

                value = str(predicted_user)

            else:

                value = "Waiting"

            self._status_card(
                "Model Prediction",
                value,
                "🧠"
            )

    # ==========================================================
    # STATUS CARD
    # ==========================================================

    def _status_card(
        self,
        label,
        value,
        icon
    ):

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">
                    {icon} {label}
                </div>

                <div class="status-value">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================================
    # SECURITY BANNER
    # ==========================================================

    def _render_security_banner(
        self,
        username,
        confidence,
        agreement,
        predicted_user
    ):

        st.markdown("---")

        if confidence is None:

            st.info(
                "🔐 **Protected session:** Login credentials "
                "are verified. Keystroke verification will "
                "be required for protected patient details."
            )

            return

        score = self._safe_confidence(
            confidence
        )

        identity_match = bool(
            predicted_user
            and username
            and str(predicted_user).lower()
            == str(username).lower()
        )

        if (
            identity_match
            and score >= 0.75
            and agreement is True
        ):

            st.success(
                "🟢 **Authentication Status: VERIFIED** — "
                "LSTM and GRU agree and the behavioral "
                "identity matches the logged-in user."
            )

        elif (
            identity_match
            and score >= 0.75
        ):

            st.warning(
                "🟡 **Authentication Status: "
                "VERIFIED WITH REVIEW** — identity matches, "
                "but LSTM/GRU agreement is not confirmed."
            )

        elif (
            score < 0.75
            or not identity_match
        ):

            st.error(
                "🔴 **Authentication Status: "
                "REQUIRES REVIEW** — protected patient "
                "access should remain locked until "
                "verification succeeds."
            )

        else:

            st.warning(
                "🟡 **Authentication Status: REVIEW**"
            )

    # ==========================================================
    # QUICK SECURITY STATUS
    # ==========================================================

    def _render_quick_status(
        self,
        confidence,
        agreement,
        predicted_user
    ):

        st.markdown(
            "### 🔐 Security Overview"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Initial Authentication",
                (
                    "Active"
                    if st.session_state.get(
                        "logged_in"
                    )
                    else "Inactive"
                )
            )

        with col2:

            st.metric(
                "Keystroke Verification",
                (
                    "Verified"
                    if confidence is not None
                    else "Pending"
                )
            )

        with col3:

            if agreement is True:

                model_value = "Agree"

            elif agreement is False:

                model_value = "Disagree"

            else:

                model_value = "Pending"

            st.metric(
                "LSTM + GRU",
                model_value
            )

        with col4:

            if predicted_user:

                st.metric(
                    "Behavioral Identity",
                    str(predicted_user)
                )

            else:

                st.metric(
                    "Behavioral Identity",
                    "Pending"
                )

    # ==========================================================
    # PATIENT SEARCH / FILTERS
    # ==========================================================

    def _get_filtered_patients(self):

        st.markdown(
            "### 🔎 Find Patient"
        )

        search_text = st.text_input(
            "Search patient records",
            placeholder=(
                "Patient name, condition, doctor, "
                "hospital, medication..."
            ),
            key="healthcare_search"
        )

        col1, col2, col3 = st.columns(3)

        # ------------------------------------------------------
        # MEDICAL CONDITION
        # ------------------------------------------------------

        with col1:

            conditions = [
                "All"
            ]

            try:

                conditions += (
                    self.patient_service
                    .get_conditions()
                )

            except Exception:

                pass

            selected_condition = st.selectbox(
                "Medical Condition",
                conditions,
                key="healthcare_condition_filter"
            )

        # ------------------------------------------------------
        # HOSPITAL
        # ------------------------------------------------------

        with col2:

            hospitals = [
                "All"
            ]

            try:

                hospitals += (
                    self.patient_service
                    .get_hospitals()
                )

            except Exception:

                pass

            selected_hospital = st.selectbox(
                "Hospital",
                hospitals,
                key="healthcare_hospital_filter"
            )

        # ------------------------------------------------------
        # ADMISSION TYPE
        # ------------------------------------------------------

        with col3:

            admission_types = [
                "All"
            ]

            try:

                admission_types += (
                    self.patient_service
                    .get_admission_types()
                )

            except Exception:

                pass

            selected_admission_type = st.selectbox(
                "Admission Type",
                admission_types,
                key="healthcare_admission_filter"
            )

        # ------------------------------------------------------
        # LOAD PATIENTS
        # ------------------------------------------------------

        try:

            if search_text.strip():

                patients = (
                    self.patient_service
                    .search_patients(
                        search_text.strip()
                    )
                )

            else:

                patients = (
                    self.patient_service
                    .get_all_patients()
                )

        except Exception as error:

            st.error(
                f"Unable to load patient records: {error}"
            )

            return []

        # ------------------------------------------------------
        # CONDITION FILTER
        # ------------------------------------------------------

        if selected_condition != "All":

            patients = [

                patient

                for patient in patients

                if str(
                    patient.get(
                        "diagnosis",
                        ""
                    )
                )
                == str(
                    selected_condition
                )

            ]

        # ------------------------------------------------------
        # HOSPITAL FILTER
        # ------------------------------------------------------

        if selected_hospital != "All":

            patients = [

                patient

                for patient in patients

                if str(
                    patient.get(
                        "hospital",
                        ""
                    )
                )
                == str(
                    selected_hospital
                )

            ]

        # ------------------------------------------------------
        # ADMISSION TYPE FILTER
        # ------------------------------------------------------

        if selected_admission_type != "All":

            patients = [

                patient

                for patient in patients

                if str(
                    patient.get(
                        "admission_type",
                        ""
                    )
                )
                == str(
                    selected_admission_type
                )

            ]

        return patients

    # ==========================================================
    # PATIENT RECORDS
    # ==========================================================

    def _render_patient_records(
        self,
        patients,
        username,
        role
    ):

        if not patients:

            st.info(
                "No patient records match the "
                "selected search or filters."
            )

            return

        st.success(
            f"Showing **{len(patients)}** "
            f"matching patient record(s)."
        )

        # ------------------------------------------------------
        # PAGINATION
        # ------------------------------------------------------

        page_size = 6

        total_pages = max(
            1,
            math.ceil(
                len(patients)
                / page_size
            )
        )

        current_page = int(
            st.session_state.get(
                "healthcare_page",
                1
            )
        )

        current_page = min(
            max(
                current_page,
                1
            ),
            total_pages
        )

        st.session_state[
            "healthcare_page"
        ] = current_page

        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )

        with col1:

            if st.button(
                "⬅ Previous",
                disabled=(
                    current_page <= 1
                ),
                key="healthcare_previous",
                use_container_width=True
            ):

                st.session_state[
                    "healthcare_page"
                ] = current_page - 1

                st.rerun()

        with col2:

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding-top:8px;
                ">
                    <b>
                        Page {current_page}
                        of {total_pages}
                    </b>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            if st.button(
                "Next ➡",
                disabled=(
                    current_page >= total_pages
                ),
                key="healthcare_next",
                use_container_width=True
            ):

                st.session_state[
                    "healthcare_page"
                ] = current_page + 1

                st.rerun()

        start = (
            current_page - 1
        ) * page_size

        page_patients = patients[
            start:
            start + page_size
        ]

        st.caption(
            f"Records {start + 1}–"
            f"{min(start + page_size, len(patients))} "
            f"of {len(patients)}"
        )

        for patient in page_patients:

            self._render_patient_card(
                patient,
                username,
                role
            )

    # ==========================================================
    # PATIENT CARD
    # ==========================================================

    def _render_patient_card(
        self,
        patient,
        username,
        role
    ):

        patient_id = patient.get(
            "patient_id",
            "Unknown"
        )

        patient_name = patient.get(
            "patient_name",
            "Unknown"
        )

        status = patient.get(
            "status",
            "Unknown"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="patient-card">',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(
                [2.1, 3.2, 1.25]
            )

            # --------------------------------------------------
            # PATIENT INFORMATION
            # --------------------------------------------------

            with col1:

                st.markdown(
                    f"""
                    <div class="patient-name">
                        👤 {patient_name}
                    </div>

                    <div class="patient-id">
                        Patient ID: {patient_id}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    f"🎂 **Age:** "
                    f"{patient.get('age', 'N/A')}"
                )

                st.write(
                    f"⚥ **Gender:** "
                    f"{patient.get('gender', 'N/A')}"
                )

                st.write(
                    f"🩸 **Blood Group:** "
                    f"{patient.get('blood_group', 'N/A')}"
                )

            # --------------------------------------------------
            # MEDICAL INFORMATION
            # --------------------------------------------------

            with col2:

                st.write(
                    f"🩺 **Condition:** "
                    f"{patient.get('diagnosis', 'N/A')}"
                )

                st.write(
                    f"👨‍⚕️ **Doctor:** "
                    f"{patient.get('doctor', 'N/A')}"
                )

                st.write(
                    f"🏥 **Hospital:** "
                    f"{patient.get('hospital', 'N/A')}"
                )

                st.write(
                    f"💊 **Medication:** "
                    f"{patient.get('medication', 'N/A')}"
                )

                if status == "Under Treatment":

                    st.warning(
                        "🟡 Under Treatment"
                    )

                elif status == "Discharged":

                    st.info(
                        "🔵 Discharged"
                    )

                else:

                    st.write(
                        f"**Status:** {status}"
                    )

            # --------------------------------------------------
            # VIEW / EDIT
            # --------------------------------------------------

            with col3:

                st.caption(
                    "Protected record"
                )

                if st.button(
                    "🔐 View",
                    key=f"view_details_{patient_id}",
                    use_container_width=True
                ):

                    self._request_patient_verification(
                        patient,
                        username
                    )

                if st.button(
                    "✏️ Edit",
                    key=f"edit_details_{patient_id}",
                    use_container_width=True
                ):

                    self._request_patient_verification(
                        patient,
                        username,
                        edit_mode=True
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    # ==========================================================
    # PATIENT ACCESS REQUEST
    # ==========================================================

    def _request_patient_verification(
        self,
        patient,
        username,
        edit_mode=False
    ):

        patient_id = patient.get(
            "patient_id"
        )

        st.session_state[
            "pending_patient_id"
        ] = patient_id

        st.session_state[
            "pending_patient_doctor"
        ] = username

        st.session_state[
            "selected_patient"
        ] = patient

        st.session_state[
            "patient_verification_required"
        ] = True

        st.session_state[
            "keystroke_verification_mode"
        ] = "patient_access"

        st.session_state[
            "patient_access_action"
        ] = (
            "edit"
            if edit_mode
            else "view"
        )

        st.session_state.pop(
            "verified_patient",
            None
        )

        st.session_state[
            "page"
        ] = "Keystroke"

        st.rerun()

    # ==========================================================
    # VERIFIED PATIENT DETAILS
    # ==========================================================

    def _render_verified_patient(self):

        patient = st.session_state.get(
            "verified_patient"
        )

        if not patient:
            return

        action = st.session_state.get(
            "patient_access_action",
            "view"
        )

        st.markdown("---")

        st.markdown(
            "## 🔓 Protected Patient Record"
        )

        st.success(
            "✅ Keystroke authentication passed. "
            "Protected patient information is unlocked."
        )

        if action == "edit":

            st.info(
                "✏️ Edit mode requested. The current "
                "PatientService is read-only, so this "
                "dashboard displays the verified record "
                "without changing the dataset."
            )

        else:

            st.caption(
                "View mode — record was unlocked "
                "after behavioral verification."
            )

        # ------------------------------------------------------
        # PATIENT INFORMATION
        # ------------------------------------------------------

        st.markdown(
            "### 👤 Patient Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            self._detail(
                "Patient ID",
                patient.get(
                    "patient_id"
                )
            )

            self._detail(
                "Patient Name",
                patient.get(
                    "patient_name"
                )
            )

            self._detail(
                "Age",
                patient.get(
                    "age"
                )
            )

            self._detail(
                "Gender",
                patient.get(
                    "gender"
                )
            )

        with col2:

            self._detail(
                "Blood Group",
                patient.get(
                    "blood_group"
                )
            )

            self._detail(
                "Status",
                patient.get(
                    "status"
                )
            )

            self._detail(
                "Admission Type",
                patient.get(
                    "admission_type"
                )
            )

            self._detail(
                "Admission Date",
                patient.get(
                    "date_of_admission"
                )
            )

        with col3:

            self._detail(
                "Doctor",
                patient.get(
                    "doctor"
                )
            )

            self._detail(
                "Hospital",
                patient.get(
                    "hospital"
                )
            )

            self._detail(
                "Room Number",
                patient.get(
                    "room_number"
                )
            )

            self._detail(
                "Discharge Date",
                patient.get(
                    "discharge_date"
                )
            )

        # ------------------------------------------------------
        # MEDICAL INFORMATION
        # ------------------------------------------------------

        st.markdown(
            "### 🩺 Medical Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            self._detail(
                "Medical Condition",
                patient.get(
                    "medical_condition",
                    patient.get(
                        "diagnosis"
                    )
                )
            )

            self._detail(
                "Medication",
                patient.get(
                    "medication"
                )
            )

            self._detail(
                "Test Results",
                patient.get(
                    "test_results"
                )
            )

        with col2:

            self._detail(
                "Insurance Provider",
                patient.get(
                    "insurance_provider"
                )
            )

            billing = patient.get(
                "billing_amount"
            )

            if (
                billing is None
                or billing == ""
            ):

                self._detail(
                    "Billing Amount",
                    "N/A"
                )

            else:

                try:

                    self._detail(
                        "Billing Amount",
                        f"${float(billing):,.2f}"
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    self._detail(
                        "Billing Amount",
                        billing
                    )

        st.markdown("---")

        if st.button(
            "🔒 Lock Patient Record",
            use_container_width=True,
            key="lock_patient_record"
        ):

            self._clear_patient_access()

            st.rerun()

    # ==========================================================
    # DETAIL HELPER
    # ==========================================================

    @staticmethod
    def _detail(
        label,
        value
    ):

        if (
            value is None
            or value == ""
        ):

            value = "N/A"

        st.write(
            f"**{label}:** {value}"
        )

    # ==========================================================
    # CLEAR PATIENT ACCESS
    # ==========================================================

    def _clear_patient_access(self):

        for key in (
            "verified_patient",
            "pending_patient_id",
            "pending_patient_doctor",
            "selected_patient",
            "patient_access_action",
        ):

            st.session_state.pop(
                key,
                None
            )

        st.session_state[
            "patient_verification_required"
        ] = False

        st.session_state[
            "keystroke_verification_mode"
        ] = None

        st.session_state[
            "page"
        ] = "Dashboard"

    # ==========================================================
    # AGENTIC AI SECURITY PANEL
    # ==========================================================

    def _render_ai_security_panel(
        self,
        confidence,
        agreement,
        predicted_user,
        username
    ):

        st.markdown("---")

        st.markdown(
            "## 🤖 Agentic AI Security Decision"
        )

        st.caption(
            "The dashboard presents the current "
            "behavioral-authentication state. The existing "
            "keystroke verification service remains "
            "responsible for the actual verification flow."
        )

        col1, col2 = st.columns(2)

        # ------------------------------------------------------
        # LSTM + GRU
        # ------------------------------------------------------

        with col1:

            st.markdown(
                '<div class="security-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🧠 LSTM + GRU"
            )

            if confidence is None:

                st.info(
                    "No behavioral verification result yet."
                )

            else:

                score = self._safe_confidence(
                    confidence
                )

                st.progress(
                    score
                )

                st.write(
                    f"**Overall confidence:** "
                    f"{score * 100:.2f}%"
                )

                if agreement is True:

                    st.success(
                        "✅ LSTM and GRU agree."
                    )

                elif agreement is False:

                    st.warning(
                        "⚠️ LSTM and GRU disagree."
                    )

                else:

                    st.info(
                        "Model agreement is not available."
                    )

                st.write(
                    f"**Expected user:** "
                    f"{username}"
                )

                st.write(
                    f"**Predicted user:** "
                    f"{predicted_user or 'N/A'}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        # ------------------------------------------------------
        # ACCESS DECISION
        # ------------------------------------------------------

        with col2:

            st.markdown(
                '<div class="security-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🛡️ Access Decision"
            )

            decision, explanation = (
                self._get_decision(
                    username,
                    confidence,
                    agreement,
                    predicted_user
                )
            )

            if decision == "ALLOW":

                st.success(
                    "🟢 **ALLOW**"
                )

            elif decision == "ALERT":

                st.warning(
                    "🟡 **ALERT / REVIEW**"
                )

            elif decision == "BLOCK":

                st.error(
                    "🔴 **BLOCK**"
                )

            else:

                st.info(
                    "⚪ **PENDING**"
                )

            st.write(
                explanation
            )

            st.caption(
                "Normal → allow/monitor | "
                "Suspicious → alert/review | "
                "Low confidence or identity mismatch "
                "→ block protected access"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    # ==========================================================
    # ACCESS DECISION LOGIC
    # ==========================================================

    @staticmethod
    def _get_decision(
        username,
        confidence,
        agreement,
        predicted_user
    ):

        if confidence is None:

            return (
                "PENDING",
                "Complete keystroke verification "
                "before making a protected-access decision."
            )

        score = Dashboard._safe_confidence(
            confidence
        )

        identity_match = bool(
            predicted_user
            and username
            and str(predicted_user).lower()
            == str(username).lower()
        )

        if (
            not identity_match
            or score < 0.75
        ):

            return (
                "BLOCK",
                "Behavioral identity or confidence "
                "is insufficient for protected access."
            )

        if agreement is False:

            return (
                "ALERT",
                "The behavioral identity matches, "
                "but LSTM and GRU disagree. "
                "Review is recommended."
            )

        if (
            score >= 0.90
            and agreement is True
        ):

            return (
                "ALLOW",
                "High-confidence behavioral "
                "authentication with model agreement."
            )

        return (
            "ALERT",
            "Authentication is plausible, but "
            "additional monitoring/review is recommended."
        )

    # ==========================================================
    # NOTIFICATIONS
    # ==========================================================

    def _render_notifications(
        self,
        username
    ):

        notification_service = (
            st.session_state.get(
                "notification_service"
            )
        )

        if notification_service is None:
            return

        try:

            notifications = (
                notification_service
                .get_notifications()
            )

        except Exception:

            notifications = []

        st.markdown("---")

        st.markdown(
            "## 🔔 Security Alerts"
        )

        if not notifications:

            st.success(
                "✅ No authentication alerts."
            )

            return

        recent_notifications = (
            notifications[-5:]
        )

        for index, notification in enumerate(
            recent_notifications
        ):

            notification_type = (
                notification.get(
                    "type",
                    ""
                )
            )

            if notification_type == (
                "patient_access_success"
            ):

                st.success(
                    f"🔓 Patient access verified "
                    f"for {notification.get('patient_name', 'patient')} "
                    f"by {notification.get('username', username)}."
                )

            elif notification_type == (
                "patient_access_failed"
            ):

                st.error(
                    f"🔒 Patient access denied "
                    f"for {notification.get('patient_name', 'patient')}."
                )

            elif notification_type == (
                "authentication_check"
            ):

                st.warning(
                    f"⚠️ Authentication check: "
                    f"{notification.get('username', username)}"
                )

            else:

                st.info(
                    "ℹ️ Security event recorded."
                )

            if not notification.get(
                "read",
                False
            ):

                if st.button(
                    "✓ Mark as Checked",
                    key=(
                        f"dashboard_check_notification_"
                        f"{index}"
                    )
                ):

                    try:

                        all_notifications = (
                            notification_service
                            .get_notifications()
                        )

                        actual_index = max(
                            0,
                            len(all_notifications)
                            - len(recent_notifications)
                            + index
                        )

                        notification_service.mark_as_read(
                            actual_index
                        )

                    except Exception:

                        pass

                    st.rerun()

    # ==========================================================
    # SYSTEM STATUS
    # ==========================================================

    def _render_system_status(self):

        st.markdown("---")

        st.markdown(
            "## ⚙️ Security System Status"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.success(
                "✅ Login"
            )

        with col2:

            st.success(
                "✅ Patient Database"
            )

        with col3:

            st.success(
                "✅ LSTM + GRU"
            )

        with col4:

            st.success(
                "✅ Agentic AI"
            )

        st.caption(
            "Healthcare records and the keystroke benchmark "
            "remain logically separated: patient data is used "
            "for healthcare access, while keystroke data is "
            "used for behavioral authentication."
        )

    # ==========================================================
    # SAFE CONFIDENCE
    # ==========================================================

    @staticmethod
    def _safe_confidence(
        value
    ):

        try:

            return min(
                max(
                    float(value),
                    0.0
                ),
                1.0
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    Dashboard().render(
        username="testuser",
        confidence=None,
        agreement=None,
        role="Doctor"
    )