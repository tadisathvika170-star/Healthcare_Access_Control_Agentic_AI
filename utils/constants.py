"""
constants.py

Project-wide constants used throughout the
Healthcare Access Control Agentic AI system.
"""

# ==========================================================
# Application
# ==========================================================

APP_NAME = "Healthcare Access Control"

APP_VERSION = "1.0.0"

ORGANIZATION = "Healthcare AI"

# ==========================================================
# User Roles
# ==========================================================

ADMIN = "Admin"

DOCTOR = "Doctor"

NURSE = "Nurse"

PATIENT = "Patient"

ROLES = [
    ADMIN,
    DOCTOR,
    NURSE,
    PATIENT
]

# ==========================================================
# Authentication
# ==========================================================

MIN_PASSWORD_LENGTH = 6

MAX_LOGIN_ATTEMPTS = 3

SESSION_TIMEOUT = 30

# ==========================================================
# Agent Decisions
# ==========================================================

ALLOW = "ALLOW"

DENY = "DENY"

VERIFY = "VERIFY"

# ==========================================================
# Risk Levels
# ==========================================================

LOW_RISK = "LOW"

MEDIUM_RISK = "MEDIUM"

HIGH_RISK = "HIGH"

# ==========================================================
# Confidence Thresholds
# ==========================================================

HIGH_CONFIDENCE = 0.90

MEDIUM_CONFIDENCE = 0.75

LOW_CONFIDENCE = 0.60

# ==========================================================
# Logging
# ==========================================================

LOGIN = "LOGIN"

LOGOUT = "LOGOUT"

REGISTER = "REGISTER"

ACCESS_GRANTED = "ACCESS_GRANTED"

ACCESS_DENIED = "ACCESS_DENIED"

FAILED_LOGIN = "FAILED_LOGIN"

# ==========================================================
# Model Names
# ==========================================================

LSTM = "LSTM"

GRU = "GRU"

FUSION = "FUSION"

# ==========================================================
# Dashboard
# ==========================================================

DASHBOARD_TITLE = "Healthcare Dashboard"

PROFILE_TITLE = "User Profile"

AUDIT_TITLE = "Audit Logs"

SETTINGS_TITLE = "Settings"

# ==========================================================
# Messages
# ==========================================================

SUCCESS_LOGIN = "Login Successful."

SUCCESS_REGISTER = "Registration Successful."

INVALID_CREDENTIALS = "Invalid Username or Password."

USER_EXISTS = "Username already exists."

ACCESS_BLOCKED = "Access Denied."

MODEL_NOT_FOUND = "Model file not found."

DATABASE_ERROR = "Database connection failed."

UNKNOWN_ERROR = "Unknown error occurred."