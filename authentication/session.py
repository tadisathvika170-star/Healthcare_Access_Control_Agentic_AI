from datetime import datetime, timedelta
from config import SESSION_TIMEOUT


class SessionManager:
    """
    Manages user login sessions.
    """

    def __init__(self):
        self.current_user = None
        self.login_time = None

    def start_session(self, user):
        """
        Start a new user session.
        """
        self.current_user = user
        self.login_time = datetime.now()

    def end_session(self):
        """
        End the current session.
        """
        self.current_user = None
        self.login_time = None

    def is_logged_in(self):
        """
        Check if a user is currently logged in.
        """
        if self.current_user is None:
            return False

        if self.is_session_expired():
            self.end_session()
            return False

        return True

    def is_session_expired(self):
        """
        Check whether the current session has expired.
        """
        if self.login_time is None:
            return True

        expiry_time = self.login_time + timedelta(minutes=SESSION_TIMEOUT)
        return datetime.now() > expiry_time

    def get_current_user(self):
        """
        Return the logged-in user's information.
        """
        return self.current_user