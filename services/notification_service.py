"""
notification_service.py

Handles notifications for authentication events.
"""

from datetime import datetime


class NotificationService:
    """
    Notification service for authentication events.
    """

    def __init__(self):
        self.notifications = []

    # ======================================================
    # Authentication Successful
    # ======================================================

    def success(self, username):

        notification = {
            "type": "success",
            "title": "Authentication Successful",
            "message": f"Welcome, {username}!",
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "read": False
        }

        self.notifications.append(notification)

        return notification

    # ======================================================
    # Authentication Failed
    # ======================================================

    def failure(self, username, reason):

        notification = {
            "type": "error",
            "title": "Authentication Failed",
            "message": f"{username}: {reason}",
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "read": False
        }

        self.notifications.append(notification)

        return notification

    # ======================================================
    # Warning
    # ======================================================

    def warning(self, message):

        notification = {
            "type": "warning",
            "title": "Warning",
            "message": message,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "read": False
        }

        self.notifications.append(notification)

        return notification

    # ======================================================
    # KEYSTROKE VERIFICATION NOTIFICATION
    # ======================================================

    def authentication_check(
        self,
        username
    ):

        notification = {

            "type": "authentication_check",

            "title":
                "🔔 Authentication Verification Required",

            "message":
                "Please check whether the user is "
                "authenticated or not.",

            "username":
                username,

            "predicted_user":
                None,

            "confidence":
                None,

            "agreement":
                None,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "read":
                False

        }

        self.notifications.append(
            notification
        )

        return notification

    # ======================================================
    # Update Keystroke Result
    # ======================================================

    def update_authentication_result(
        self,
        username,
        predicted_user,
        confidence,
        agreement
    ):

        # Find latest authentication notification
        for notification in reversed(
            self.notifications
        ):

            if (
                notification["type"]
                == "authentication_check"
                and notification["username"]
                == username
            ):

                notification[
                    "predicted_user"
                ] = predicted_user

                notification[
                    "confidence"
                ] = confidence

                notification[
                    "agreement"
                ] = agreement

                return notification

        # If notification doesn't exist,
        # create one automatically.
        notification = self.authentication_check(
            username
        )

        notification[
            "predicted_user"
        ] = predicted_user

        notification[
            "confidence"
        ] = confidence

        notification[
            "agreement"
        ] = agreement

        return notification

    # ======================================================
    # Get All Notifications
    # ======================================================

    def get_notifications(self):

        return self.notifications

    # ======================================================
    # Unread Notifications
    # ======================================================

    def get_unread_notifications(self):

        return [
            notification
            for notification in self.notifications
            if not notification["read"]
        ]

    # ======================================================
    # Mark Notification As Read
    # ======================================================

    def mark_as_read(self, index):

        if (
            0 <= index
            < len(self.notifications)
        ):

            self.notifications[index][
                "read"
            ] = True

    # ======================================================
    # Mark All As Read
    # ======================================================

    def mark_all_as_read(self):

        for notification in self.notifications:

            notification["read"] = True


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    notifier = NotificationService()

    print("=" * 60)
    print("Notification Service Test")
    print("=" * 60)

    result = notifier.authentication_check(
        "s041"
    )

    print("\nAuthentication Check:")
    print(result)

    notifier.update_authentication_result(
        username="s041",
        predicted_user="s041",
        confidence=0.94,
        agreement=True
    )

    print("\nUpdated Notification:")
    print(
        notifier.get_notifications()
    )