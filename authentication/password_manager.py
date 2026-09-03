import bcrypt


class PasswordManager:
    """
    Handles password hashing and verification.
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a plain-text password.
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a plain-text password against a hashed password.
        """
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )