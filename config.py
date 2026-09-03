from pathlib import Path

# ======================================================
# PROJECT INFORMATION
# ======================================================

PROJECT_NAME = "Healthcare Access Control Agentic AI"

BASE_DIR = Path(__file__).resolve().parent


# ======================================================
# DIRECTORIES
# ======================================================

ASSETS_DIR = BASE_DIR / "assets"

DATABASE_DIR = BASE_DIR / "database"

DATASET_DIR = BASE_DIR / "dataset"

MODELS_DIR = BASE_DIR / "models"

LOGS_DIR = BASE_DIR / "logs"

REPORTS_DIR = BASE_DIR / "reports"


# ======================================================
# DATABASE
# ======================================================

# User authentication database
DATABASE_PATH = DATABASE_DIR / "users.db"

# Security and authentication audit database
AUDIT_DATABASE_PATH = DATABASE_DIR / "audit_logs.db"

# Healthcare database
HEALTHCARE_DATABASE_PATH = DATABASE_DIR / "healthcare.db"


# ======================================================
# DATASETS
# ======================================================

# ------------------------------------------------------
# Keystroke Dynamics Dataset
# Used only for LSTM + GRU authentication
# ------------------------------------------------------

KEYSTROKE_DATASET_PATH = (
    DATASET_DIR / "DSL-StrongPasswordData.csv"
)

# Backward compatibility with existing code
DATASET_PATH = KEYSTROKE_DATASET_PATH


# ------------------------------------------------------
# Healthcare Dataset
# Used for patient and medical records
# ------------------------------------------------------

HEALTHCARE_DATASET_PATH = (
    DATASET_DIR / "healthcare_dataset.csv"
)


# ======================================================
# MODEL FILES
# ======================================================

LSTM_MODEL_PATH = (
    MODELS_DIR / "lstm_model.pth"
)

GRU_MODEL_PATH = (
    MODELS_DIR / "gru_model.pth"
)

SCALER_PATH = (
    MODELS_DIR / "scaler.pkl"
)

LABEL_ENCODER_PATH = (
    MODELS_DIR / "label_encoder.pkl"
)


# ======================================================
# RANDOM SEED
# ======================================================

RANDOM_STATE = 42


# ======================================================
# AUTHENTICATION SETTINGS
# ======================================================

MAX_LOGIN_ATTEMPTS = 3

SESSION_TIMEOUT = 30


# ======================================================
# AGENTIC AI SETTINGS
# ======================================================

HIGH_CONFIDENCE = 0.95

MEDIUM_CONFIDENCE = 0.80

LOW_CONFIDENCE = 0.60


# ======================================================
# HEALTHCARE SETTINGS
# ======================================================

# Number of records to load/display by default
HEALTHCARE_PAGE_SIZE = 20

# Enable healthcare dataset integration
HEALTHCARE_DATASET_ENABLED = True