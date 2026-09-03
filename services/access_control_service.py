"""
access_control_service.py

Determines the level of access granted to a user
after successful authentication.
"""


class AccessControlService:
    """
    Access control based on authentication result
    and confidence score.
    """

    def __init__(self):

        self.minimum_confidence = 0.80

    def evaluate(self, authentication_result):
        """
        Decide whether access should be granted.

        Parameters
        ----------
        authentication_result : dict

        Returns
        -------
        dict
        """

        if not authentication_result["status"]:

            return {
                "access": False,
                "message": authentication_result["message"]
            }

        details = authentication_result["details"]

        confidence = details["confidence"]

        if confidence >= self.minimum_confidence:

            return {
                "access": True,
                "role": "Authorized User",
                "confidence": confidence,
                "message": "Access Granted"
            }

        return {
            "access": False,
            "role": "Unknown",
            "confidence": confidence,
            "message": "Access Denied (Low Confidence)"
        }


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    service = AccessControlService()

    sample = {
        "status": True,
        "details": {
            "confidence": 0.92
        }
    }

    result = service.evaluate(sample)

    print("=" * 60)
    print("Access Control Test")
    print("=" * 60)
    print(result)