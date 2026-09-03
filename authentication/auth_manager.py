from config import DATABASE_PATH
from database.database import DatabaseManager
from authentication.password_manager import PasswordManager


class AuthManager:
    """
    Handles user registration and login.
    """

    def __init__(self):
        self.db = DatabaseManager(DATABASE_PATH)

    def register_user(self, username, password, full_name, role):
        """
        Register a new user.
        """

        # Check if username already exists
        existing_user = self.db.fetchone(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        if existing_user:
            return False, "Username already exists."

        hashed_password = PasswordManager.hash_password(password)

        self.db.execute(
            """
            INSERT INTO users (username, password, full_name, role)
            VALUES (?, ?, ?, ?)
            """,
            (username, hashed_password, full_name, role)
        )

        return True, "Registration successful."

    def login_user(self, username, password):
        """
        Verify login credentials.
        """

        user = self.db.fetchone(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        if user is None:
            return False, "User not found."

        if PasswordManager.verify_password(password, user["password"]):
            return True, dict(user)

        return False, "Invalid password."