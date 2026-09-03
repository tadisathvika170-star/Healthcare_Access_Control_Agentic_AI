"""
Credential Agent

This agent verifies the user's username and password.
"""

from authentication.auth_manager import AuthManager


class CredentialAgent:
    """
    Responsible for credential verification.
    """

    def __init__(self):
        self.auth_manager = AuthManager()

    def authenticate(self, username: str, password: str):
        """
        Verify username and password.

        Returns
        -------
        dict
        {
            "success": bool,
            "message": str,
            "user": dict | None
        }
        """

        success, result = self.auth_manager.login_user(username, password)

        if success:
            return {
                "success": True,
                "message": "Credential verification successful.",
                "user": result
            }

        return {
            "success": False,
            "message": result,
            "user": None
        }

    def register(self, username, password, full_name, role):
        """
        Register a new user.
        """

        success, message = self.auth_manager.register_user(
            username,
            password,
            full_name,
            role
        )

        return {
            "success": success,
            "message": message
        }