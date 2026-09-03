"""
patient_service.py

Healthcare patient data service.

This module loads patient and medical records from
healthcare_dataset.csv instead of using hard-coded
patient records.

The keystroke dataset remains separate and is used
only for LSTM/GRU authentication.
"""

from pathlib import Path
import sys

import pandas as pd


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

from config import HEALTHCARE_DATASET_PATH


class PatientService:
    """
    Service for loading and accessing healthcare records.

    The service reads healthcare_dataset.csv and converts
    every row into a dictionary that can be used by the
    existing Streamlit frontend.
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self, dataset_path=None):

        if dataset_path is not None:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = Path(
                HEALTHCARE_DATASET_PATH
            )

        self.dataframe = pd.DataFrame()

        self.patients = []

        self._load_dataset()

    # ======================================================
    # LOAD DATASET
    # ======================================================

    def _load_dataset(self):
        """
        Load the healthcare CSV file.
        """

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                "\nHealthcare dataset not found.\n"
                f"Expected location:\n"
                f"{self.dataset_path}\n\n"
                "Please make sure that "
                "'healthcare_dataset.csv' is inside "
                "the dataset folder."
            )

        try:

            dataframe = pd.read_csv(
                self.dataset_path
            )

        except Exception as error:

            raise RuntimeError(
                "Unable to read healthcare dataset: "
                f"{error}"
            ) from error

        # --------------------------------------------------
        # Required columns
        # --------------------------------------------------

        required_columns = [

            "Name",

            "Age",

            "Gender",

            "Blood Type",

            "Medical Condition",

            "Date of Admission",

            "Doctor",

            "Hospital",

            "Insurance Provider",

            "Billing Amount",

            "Room Number",

            "Admission Type",

            "Discharge Date",

            "Medication",

            "Test Results",
        ]

        missing_columns = [

            column
            for column in required_columns
            if column not in dataframe.columns

        ]

        if missing_columns:

            raise ValueError(
                "\nHealthcare dataset is missing "
                "the following required columns:\n"
                f"{missing_columns}"
            )

        # --------------------------------------------------
        # Clean text columns
        # --------------------------------------------------

        text_columns = [

            "Name",

            "Gender",

            "Blood Type",

            "Medical Condition",

            "Doctor",

            "Hospital",

            "Insurance Provider",

            "Admission Type",

            "Medication",

            "Test Results",
        ]

        for column in text_columns:

            dataframe[column] = (

                dataframe[column]

                .fillna("")

                .astype(str)

                .str.strip()
            )

        # --------------------------------------------------
        # Clean numerical columns
        # --------------------------------------------------

        dataframe["Age"] = pd.to_numeric(
            dataframe["Age"],
            errors="coerce"
        )

        dataframe["Billing Amount"] = pd.to_numeric(
            dataframe["Billing Amount"],
            errors="coerce"
        )

        dataframe["Room Number"] = pd.to_numeric(
            dataframe["Room Number"],
            errors="coerce"
        )

        # --------------------------------------------------
        # Convert dates
        # --------------------------------------------------

        dataframe["Date of Admission"] = pd.to_datetime(
            dataframe["Date of Admission"],
            errors="coerce"
        )

        dataframe["Discharge Date"] = pd.to_datetime(
            dataframe["Discharge Date"],
            errors="coerce"
        )

        # --------------------------------------------------
        # Remove records without patient names
        # --------------------------------------------------

        dataframe = dataframe[
            dataframe["Name"].str.strip() != ""
        ].copy()

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        # --------------------------------------------------
        # Save dataframe
        # --------------------------------------------------

        self.dataframe = dataframe

        # --------------------------------------------------
        # Convert every row into application record
        # --------------------------------------------------

        self.patients = []

        for index, row in dataframe.iterrows():

            patient = self._convert_row_to_patient(
                row,
                index
            )

            self.patients.append(
                patient
            )

    # ======================================================
    # CONVERT ROW TO PATIENT
    # ======================================================

    def _convert_row_to_patient(
        self,
        row,
        index
    ):
        """
        Convert one healthcare dataset row into the
        format expected by the application.
        """

        # --------------------------------------------------
        # Generate patient ID
        # --------------------------------------------------

        patient_id = (
            f"P{index + 1:05d}"
        )

        # --------------------------------------------------
        # Admission date
        # --------------------------------------------------

        admission_date = self._format_date(
            row.get(
                "Date of Admission"
            )
        )

        # --------------------------------------------------
        # Discharge date
        # --------------------------------------------------

        discharge_date = self._format_date(
            row.get(
                "Discharge Date"
            )
        )

        # --------------------------------------------------
        # Patient status
        # --------------------------------------------------

        status = self._calculate_status(
            discharge_date
        )

        # --------------------------------------------------
        # Create patient dictionary
        # --------------------------------------------------

        patient = {

            # ==============================================
            # Existing application fields
            # ==============================================

            "patient_id":
                patient_id,

            "patient_name":
                self._safe_string(
                    row.get("Name")
                ),

            "age":
                self._safe_value(
                    row.get("Age")
                ),

            "gender":
                self._safe_string(
                    row.get("Gender")
                ),

            "blood_group":
                self._safe_string(
                    row.get("Blood Type")
                ),

            "diagnosis":
                self._safe_string(
                    row.get(
                        "Medical Condition"
                    )
                ),

            "appointment_date":
                admission_date,

            "status":
                status,

            "doctor":
                self._safe_string(
                    row.get("Doctor")
                ),

            "doctor_username":
                self._safe_string(
                    row.get("Doctor")
                ),

            # ==============================================
            # Healthcare information
            # ==============================================

            "hospital":
                self._safe_string(
                    row.get("Hospital")
                ),

            "insurance_provider":
                self._safe_string(
                    row.get(
                        "Insurance Provider"
                    )
                ),

            "billing_amount":
                self._safe_value(
                    row.get(
                        "Billing Amount"
                    )
                ),

            "room_number":
                self._safe_value(
                    row.get(
                        "Room Number"
                    )
                ),

            "admission_type":
                self._safe_string(
                    row.get(
                        "Admission Type"
                    )
                ),

            "discharge_date":
                discharge_date,

            "medication":
                self._safe_string(
                    row.get("Medication")
                ),

            "test_results":
                self._safe_string(
                    row.get(
                        "Test Results"
                    )
                ),

            # ==============================================
            # Additional aliases
            # ==============================================

            "date_of_admission":
                admission_date,

            "medical_condition":
                self._safe_string(
                    row.get(
                        "Medical Condition"
                    )
                ),
        }

        return patient

    # ======================================================
    # SAFE STRING
    # ======================================================

    @staticmethod
    def _safe_string(value):
        """
        Safely convert a value into a string.
        """

        if value is None:
            return ""

        try:

            if pd.isna(value):
                return ""

        except Exception:
            pass

        return str(value).strip()

    # ======================================================
    # SAFE VALUE
    # ======================================================

    @staticmethod
    def _safe_value(value):
        """
        Safely convert pandas/numpy numeric values
        into normal Python values.
        """

        if value is None:
            return None

        try:

            if pd.isna(value):
                return None

        except Exception:
            pass

        try:

            if isinstance(value, float):

                if value.is_integer():

                    return int(value)

                return round(
                    value,
                    2
                )

            return int(value)

        except (
            TypeError,
            ValueError
        ):

            return value

    # ======================================================
    # FORMAT DATE
    # ======================================================

    @staticmethod
    def _format_date(value):
        """
        Convert date values into YYYY-MM-DD format.
        """

        if value is None:
            return ""

        try:

            if pd.isna(value):
                return ""

        except Exception:
            pass

        try:

            timestamp = pd.to_datetime(
                value,
                errors="coerce"
            )

            if pd.isna(timestamp):
                return ""

            return timestamp.strftime(
                "%Y-%m-%d"
            )

        except Exception:

            return str(value)

    # ======================================================
    # CALCULATE STATUS
    # ======================================================

    @staticmethod
    def _calculate_status(
        discharge_date
    ):
        """
        Determine patient status.

        The healthcare dataset does not have a status
        column.

        Therefore:

        Discharge date available
                -> Discharged

        No discharge date
                -> Under Treatment
        """

        if discharge_date:

            return "Discharged"

        return "Under Treatment"

    # ======================================================
    # GET ALL PATIENTS
    # ======================================================

    def get_all_patients(self):

        return self.patients

    # ======================================================
    # GET PATIENT COUNT
    # ======================================================

    def get_patient_count(self):

        return len(
            self.patients
        )

    # ======================================================
    # GET PATIENT BY ID
    # ======================================================

    def get_patient_by_id(
        self,
        patient_id
    ):
        """
        Find a patient using patient ID.
        """

        if not patient_id:

            return None

        patient_id = str(
            patient_id
        ).strip()

        for patient in self.patients:

            if (
                patient.get(
                    "patient_id"
                )
                == patient_id
            ):

                return patient

        return None

    # ======================================================
    # GET PATIENT
    # ======================================================

    def get_patient(
        self,
        patient_id,
        doctor_username=None
    ):
        """
        Get patient record.

        doctor_username is retained for compatibility
        with existing application code.
        """

        patient = self.get_patient_by_id(
            patient_id
        )

        if patient is None:

            return None

        return patient

    # ======================================================
    # SEARCH PATIENTS
    # ======================================================

    def search_patients(
        self,
        search_text=""
    ):
        """
        Search patients by multiple healthcare fields.
        """

        search_text = str(
            search_text
        ).strip().lower()

        if not search_text:

            return self.get_all_patients()

        results = []

        searchable_fields = [

            "patient_id",

            "patient_name",

            "diagnosis",

            "doctor",

            "hospital",

            "blood_group",

            "medication",

            "test_results",

            "admission_type",
        ]

        for patient in self.patients:

            for field in searchable_fields:

                value = str(
                    patient.get(
                        field,
                        ""
                    )
                ).lower()

                if search_text in value:

                    results.append(
                        patient
                    )

                    break

        return results

    # ======================================================
    # SEARCH BY CONDITION
    # ======================================================

    def get_patients_by_condition(
        self,
        condition
    ):
        """
        Get all patients having a particular
        medical condition.
        """

        condition = str(
            condition
        ).strip().lower()

        if not condition:

            return []

        return [

            patient

            for patient in self.patients

            if str(
                patient.get(
                    "diagnosis",
                    ""
                )
            ).strip().lower()
            == condition
        ]

    # ======================================================
    # GET DOCTOR PATIENTS
    # ======================================================

    def get_doctor_patients(
        self,
        doctor_username
    ):
        """
        Return records matching the doctor field.

        NOTE:
        Healthcare dataset doctor names and application
        login usernames are currently separate.

        A proper mapping will be implemented later
        during role-based access control integration.
        """

        if not doctor_username:

            return []

        doctor_username = str(
            doctor_username
        ).strip().lower()

        return [

            patient

            for patient in self.patients

            if str(
                patient.get(
                    "doctor",
                    ""
                )
            ).strip().lower()
            == doctor_username
        ]

    # ======================================================
    # GET UNIQUE CONDITIONS
    # ======================================================

    def get_conditions(self):

        conditions = set()

        for patient in self.patients:

            condition = patient.get(
                "diagnosis",
                ""
            )

            if condition:

                conditions.add(
                    condition
                )

        return sorted(
            conditions
        )

    # ======================================================
    # GET UNIQUE DOCTORS
    # ======================================================

    def get_doctors(self):

        doctors = set()

        for patient in self.patients:

            doctor = patient.get(
                "doctor",
                ""
            )

            if doctor:

                doctors.add(
                    doctor
                )

        return sorted(
            doctors
        )

    # ======================================================
    # GET UNIQUE HOSPITALS
    # ======================================================

    def get_hospitals(self):

        hospitals = set()

        for patient in self.patients:

            hospital = patient.get(
                "hospital",
                ""
            )

            if hospital:

                hospitals.add(
                    hospital
                )

        return sorted(
            hospitals
        )

    # ======================================================
    # GET UNIQUE MEDICATIONS
    # ======================================================

    def get_medications(self):

        medications = set()

        for patient in self.patients:

            medication = patient.get(
                "medication",
                ""
            )

            if medication:

                medications.add(
                    medication
                )

        return sorted(
            medications
        )

    # ======================================================
    # GET UNIQUE BLOOD GROUPS
    # ======================================================

    def get_blood_groups(self):

        blood_groups = set()

        for patient in self.patients:

            blood_group = patient.get(
                "blood_group",
                ""
            )

            if blood_group:

                blood_groups.add(
                    blood_group
                )

        return sorted(
            blood_groups
        )

    # ======================================================
    # GET UNIQUE ADMISSION TYPES
    # ======================================================

    def get_admission_types(self):

        admission_types = set()

        for patient in self.patients:

            admission_type = patient.get(
                "admission_type",
                ""
            )

            if admission_type:

                admission_types.add(
                    admission_type
                )

        return sorted(
            admission_types
        )

    # ======================================================
    # GET UNIQUE TEST RESULTS
    # ======================================================

    def get_test_results(self):

        test_results = set()

        for patient in self.patients:

            result = patient.get(
                "test_results",
                ""
            )

            if result:

                test_results.add(
                    result
                )

        return sorted(
            test_results
        )

    # ======================================================
    # GET STATISTICS
    # ======================================================

    def get_statistics(self):
        """
        Generate healthcare dataset statistics.
        """

        total_patients = len(
            self.patients
        )

        total_conditions = len(
            self.get_conditions()
        )

        total_doctors = len(
            self.get_doctors()
        )

        total_hospitals = len(
            self.get_hospitals()
        )

        total_medications = len(
            self.get_medications()
        )

        total_blood_groups = len(
            self.get_blood_groups()
        )

        discharged = sum(

            1

            for patient in self.patients

            if patient.get(
                "status"
            )
            == "Discharged"
        )

        under_treatment = sum(

            1

            for patient in self.patients

            if patient.get(
                "status"
            )
            == "Under Treatment"
        )

        return {

            "total_patients":
                total_patients,

            "total_conditions":
                total_conditions,

            "total_doctors":
                total_doctors,

            "total_hospitals":
                total_hospitals,

            "total_medications":
                total_medications,

            "total_blood_groups":
                total_blood_groups,

            "discharged":
                discharged,

            "under_treatment":
                under_treatment,
        }


# ==========================================================
# DIRECT TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("        HEALTHCARE PATIENT SERVICE TEST")
    print("=" * 70)

    try:

        # --------------------------------------------------
        # Create service
        # --------------------------------------------------

        service = PatientService()

        # --------------------------------------------------
        # Dataset information
        # --------------------------------------------------

        print()

        print(
            "Dataset path:"
        )

        print(
            service.dataset_path
        )

        print()

        print(
            "Total healthcare records:",
            service.get_patient_count()
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        statistics = (
            service.get_statistics()
        )

        print()

        print(
            "Healthcare Statistics"
        )

        print("-" * 70)

        print(
            "Total Patients       :",
            statistics[
                "total_patients"
            ]
        )

        print(
            "Medical Conditions   :",
            statistics[
                "total_conditions"
            ]
        )

        print(
            "Doctors              :",
            statistics[
                "total_doctors"
            ]
        )

        print(
            "Hospitals            :",
            statistics[
                "total_hospitals"
            ]
        )

        print(
            "Medications          :",
            statistics[
                "total_medications"
            ]
        )

        print(
            "Blood Groups         :",
            statistics[
                "total_blood_groups"
            ]
        )

        print(
            "Discharged           :",
            statistics[
                "discharged"
            ]
        )

        print(
            "Under Treatment      :",
            statistics[
                "under_treatment"
            ]
        )

        # --------------------------------------------------
        # Display first patient
        # --------------------------------------------------

        print()

        print(
            "First Healthcare Record"
        )

        print("-" * 70)

        patients = (
            service.get_all_patients()
        )

        if patients:

            first_patient = patients[0]

            for key, value in (
                first_patient.items()
            ):

                print(
                    f"{key:25}: {value}"
                )

        # --------------------------------------------------
        # Display conditions
        # --------------------------------------------------

        print()

        print(
            "Medical Conditions:"
        )

        print(
            service.get_conditions()
        )

        # --------------------------------------------------
        # Display first five doctors
        # --------------------------------------------------

        print()

        print(
            "First 5 Doctors:"
        )

        doctors = (
            service.get_doctors()
        )

        for doctor in doctors[:5]:

            print(
                f"  - {doctor}"
            )

        # --------------------------------------------------
        # Display first five hospitals
        # --------------------------------------------------

        print()

        print(
            "First 5 Hospitals:"
        )

        hospitals = (
            service.get_hospitals()
        )

        for hospital in hospitals[:5]:

            print(
                f"  - {hospital}"
            )

        # --------------------------------------------------
        # Search test
        # --------------------------------------------------

        print()

        print(
            "Search Test: Diabetes"
        )

        search_results = (
            service.search_patients(
                "Diabetes"
            )
        )

        print(
            "Matching records:",
            len(search_results)
        )

        # --------------------------------------------------
        # Successful completion
        # --------------------------------------------------

        print()

        print("=" * 70)

        print(
            "SUCCESS: Healthcare dataset "
            "loaded successfully."
        )

        print("=" * 70)

    except Exception as error:

        print()

        print("=" * 70)

        print(
            "ERROR: Healthcare dataset "
            "loading failed."
        )

        print(
            f"Reason: {error}"
        )

        print("=" * 70)

        raise