"""
helpers.py

Common helper functions used across the
Healthcare Access Control Agentic AI project.
"""

from datetime import datetime
import random
import string


def current_timestamp():
    """
    Returns current date and time.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_session_id(length=32):
    """
    Generates a random session ID.
    """

    characters = string.ascii_letters + string.digits

    return "".join(
        random.choice(characters)
        for _ in range(length)
    )


def calculate_average(values):
    """
    Returns the average of a list.
    """

    if not values:
        return 0

    return sum(values) / len(values)


def confidence_percentage(confidence):
    """
    Converts confidence score to percentage.
    """

    return round(confidence * 100, 2)


def risk_level(confidence):
    """
    Returns risk level based on confidence.
    """

    if confidence >= 0.90:
        return "LOW"

    if confidence >= 0.75:
        return "MEDIUM"

    return "HIGH"


def format_username(username):
    """
    Removes whitespace and converts to lowercase.
    """

    return username.strip().lower()


def validate_password(password):
    """
    Basic password validation.
    """

    return len(password) >= 6


def random_access_token(length=16):
    """
    Generates a random access token.
    """

    characters = string.ascii_uppercase + string.digits

    return "".join(
        random.choice(characters)
        for _ in range(length)
    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    print("Timestamp :", current_timestamp())

    print("Session ID :", generate_session_id())

    print("Average :", calculate_average([90, 95, 85]))

    print("Confidence :", confidence_percentage(0.9435))

    print("Risk :", risk_level(0.82))

    print("Username :", format_username("  S041  "))

    print("Password Valid :", validate_password("Password123"))

    print("Token :", random_access_token())