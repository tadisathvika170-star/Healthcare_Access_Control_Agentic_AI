import json
import os
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.80

AUDIT_LOG_FILE = "logs/access_log.json"


# ============================================================
# User -> Role mapping
# ============================================================
#
# IMPORTANT:
# These are DEMONSTRATION roles for the project.
# In a real healthcare system, roles must come from the
# hospital's identity/access-management system.
#
# All 51 model users are represented.
# ============================================================

USER_ROLES = {
    0: "doctor",
    1: "doctor",
    2: "doctor",
    3: "doctor",
    4: "doctor",

    5: "nurse",
    6: "nurse",
    7: "nurse",
    8: "nurse",
    9: "nurse",
    10: "nurse",
    11: "nurse",
    12: "nurse",
    13: "nurse",
    14: "nurse",

    15: "administrator",
    16: "administrator",
    17: "administrator",
    18: "administrator",
    19: "administrator",

    20: "researcher",
    21: "researcher",
    22: "researcher",
    23: "researcher",
    24: "researcher",

    25: "doctor",
    26: "doctor",
    27: "doctor",
    28: "doctor",
    29: "doctor",

    30: "nurse",
    31: "nurse",
    32: "nurse",
    33: "nurse",
    34: "nurse",

    35: "administrator",
    36: "administrator",
    37: "administrator",
    38: "administrator",
    39: "administrator",

    40: "researcher",
    41: "researcher",
    42: "researcher",
    43: "researcher",
    44: "researcher",

    45: "doctor",
    46: "nurse",
    47: "doctor",
    48: "nurse",
    49: "administrator",
    50: "researcher",
}


# ============================================================
# Role -> permissions
# ============================================================

ROLE_PERMISSIONS = {

    "doctor": [
        "patient_records",
        "clinical_records",
        "prescriptions",
    ],

    "nurse": [
        "patient_records",
        "clinical_records",
    ],

    "administrator": [
        "system_settings",
        "user_management",
        "audit_logs",
    ],

    "researcher": [
        "research_data",
        "anonymized_records",
    ],
}


# ============================================================
# Audit logging
# ============================================================

def write_audit_log(
    predicted_user,
    confidence,
    role,
    requested_resource,
    decision,
    reason
):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    event = {
        "timestamp": datetime.now().isoformat(),

        "predicted_user": int(
            predicted_user
        ),

        "confidence": round(
            float(confidence),
            4
        ),

        "role": role,

        "requested_resource":
            requested_resource,

        "decision": decision,

        "reason": reason
    }


    existing_logs = []

    if os.path.exists(AUDIT_LOG_FILE):

        try:

            with open(
                AUDIT_LOG_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                existing_logs = json.load(file)

        except (
            json.JSONDecodeError,
            FileNotFoundError
        ):

            existing_logs = []


    existing_logs.append(event)


    with open(
        AUDIT_LOG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            existing_logs,
            file,
            indent=4
        )


# ============================================================
# Authorization engine
# ============================================================

def authorize_access(
    predicted_user,
    confidence,
    requested_resource
):

    predicted_user = int(
        predicted_user
    )

    confidence = float(
        confidence
    )


    # --------------------------------------------------------
    # 1. Confidence check
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        decision = "DENY"

        reason = (
            "Identity confidence is below "
            "the required threshold."
        )

        write_audit_log(
            predicted_user,
            confidence,
            "unknown",
            requested_resource,
            decision,
            reason
        )

        return {
            "decision": decision,
            "user": predicted_user,
            "confidence": confidence,
            "role": "unknown",
            "resource": requested_resource,
            "reason": reason
        }


    # --------------------------------------------------------
    # 2. Find user role
    # --------------------------------------------------------

    role = USER_ROLES.get(
        predicted_user
    )


    if role is None:

        decision = "DENY"

        reason = (
            "User is not registered "
            "in the authorization system."
        )

        write_audit_log(
            predicted_user,
            confidence,
            "unknown",
            requested_resource,
            decision,
            reason
        )

        return {
            "decision": decision,
            "user": predicted_user,
            "confidence": confidence,
            "role": "unknown",
            "resource": requested_resource,
            "reason": reason
        }


    # --------------------------------------------------------
    # 3. Get permissions
    # --------------------------------------------------------

    permissions = ROLE_PERMISSIONS.get(
        role,
        []
    )


    # --------------------------------------------------------
    # 4. Resource authorization
    # --------------------------------------------------------

    if requested_resource in permissions:

        decision = "GRANT"

        reason = (
            f"User has the '{role}' role "
            f"and is authorized for this resource."
        )

    else:

        decision = "DENY"

        reason = (
            f"Role '{role}' does not have "
            f"permission for this resource."
        )


    # --------------------------------------------------------
    # 5. Audit event
    # --------------------------------------------------------

    write_audit_log(
        predicted_user,
        confidence,
        role,
        requested_resource,
        decision,
        reason
    )


    return {
        "decision": decision,
        "user": predicted_user,
        "confidence": confidence,
        "role": role,
        "resource": requested_resource,
        "reason": reason
    }


# ============================================================
# Display all configured users
# ============================================================

def display_users():

    print("\n" + "=" * 60)
    print("CONFIGURED HEALTHCARE USERS")
    print("=" * 60)

    for user_id, role in USER_ROLES.items():

        print(
            f"User {user_id:02d} -> {role}"
        )

    print("=" * 60)


# ============================================================
# Test authorization scenarios
# ============================================================

def run_tests():

    print("\n" + "=" * 60)
    print("HEALTHCARE ACCESS CONTROL TESTS")
    print("=" * 60)


    tests = [

        # Doctor accessing patient records
        (
            0,
            0.95,
            "patient_records"
        ),

        # Nurse accessing clinical records
        (
            5,
            0.93,
            "clinical_records"
        ),

        # Nurse attempting administrator resource
        (
            5,
            0.94,
            "system_settings"
        ),

        # Administrator accessing audit logs
        (
            15,
            0.97,
            "audit_logs"
        ),

        # Researcher accessing research data
        (
            20,
            0.91,
            "research_data"
        ),

        # Researcher attempting patient records
        (
            20,
            0.92,
            "patient_records"
        ),

        # Unknown user
        (
            99,
            0.95,
            "patient_records"
        ),

        # Low confidence
        (
            0,
            0.55,
            "patient_records"
        ),
    ]


    for user, confidence, resource in tests:

        result = authorize_access(
            predicted_user=user,
            confidence=confidence,
            requested_resource=resource
        )

        print("\n" + "-" * 60)

        print(
            f"User       : {result['user']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.2%}"
        )

        print(
            f"Role       : {result['role']}"
        )

        print(
            f"Resource   : {result['resource']}"
        )

        print(
            f"Decision   : {result['decision']}"
        )

        print(
            f"Reason     : {result['reason']}"
        )


    print("\n" + "=" * 60)

    print(
        f"Audit log: {AUDIT_LOG_FILE}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    display_users()

    run_tests()