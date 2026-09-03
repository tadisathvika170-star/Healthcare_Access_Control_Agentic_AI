"""
encryption.py

Encryption utility for the
Healthcare Access Control Agentic AI project.
"""

import base64
import hashlib
import secrets


class Encryption:
    """
    Provides hashing and encoding utilities.
    """

    @staticmethod
    def generate_salt(length=16):
        """
        Generate a random hexadecimal salt.
        """
        return secrets.token_hex(length)

    @staticmethod
    def hash_text(text):
        """
        Returns SHA-256 hash of the given text.
        """
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def hash_with_salt(text, salt):
        """
        Hash text using SHA-256 with salt.
        """
        return hashlib.sha256(
            (text + salt).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def encode(text):
        """
        Encode text using Base64.
        """
        return base64.b64encode(
            text.encode("utf-8")
        ).decode("utf-8")

    @staticmethod
    def decode(encoded_text):
        """
        Decode Base64 encoded text.
        """
        return base64.b64decode(
            encoded_text.encode("utf-8")
        ).decode("utf-8")

    @staticmethod
    def verify_hash(text, hashed_text):
        """
        Verify SHA-256 hash.
        """
        return Encryption.hash_text(text) == hashed_text


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    sample = "HealthcareAI"

    print("=" * 60)
    print("Encryption Utility Test")
    print("=" * 60)

    salt = Encryption.generate_salt()

    print("Generated Salt")
    print(salt)

    hashed = Encryption.hash_text(sample)

    print("\nSHA256 Hash")
    print(hashed)

    salted_hash = Encryption.hash_with_salt(
        sample,
        salt
    )

    print("\nSalted Hash")
    print(salted_hash)

    encoded = Encryption.encode(sample)

    print("\nEncoded")
    print(encoded)

    decoded = Encryption.decode(encoded)

    print("\nDecoded")
    print(decoded)

    print("\nHash Verification")
    print(
        Encryption.verify_hash(
            sample,
            hashed
        )
    )