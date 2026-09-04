"""
keystroke/feature_extractor.py

Live keystroke feature extraction for:

    .tie5Roanl

Produces exactly 31 features matching the
DSL-StrongPasswordData dataset.

Important:
- R is normalized to r.
- Enter is normalized to Return.
- Shift is handled separately.
- Invalid/duplicate browser events are ignored.
- Output is a NumPy array of shape (31,).
"""

from typing import Any, Dict, List, Optional

import numpy as np


# ==========================================================
# DATASET FEATURE ORDER
# ==========================================================

FEATURE_NAMES = [
    "H.period",
    "DD.period.t",
    "UD.period.t",

    "H.t",
    "DD.t.i",
    "UD.t.i",

    "H.i",
    "DD.i.e",
    "UD.i.e",

    "H.e",
    "DD.e.five",
    "UD.e.five",

    "H.five",
    "DD.five.Shift.r",
    "UD.five.Shift.r",

    "H.Shift.r",

    "DD.Shift.r.o",
    "UD.Shift.r.o",

    "H.o",
    "DD.o.a",
    "UD.o.a",

    "H.a",
    "DD.a.n",
    "UD.a.n",

    "H.n",
    "DD.n.l",
    "UD.n.l",

    "H.l",
    "DD.l.Return",
    "UD.l.Return",

    "H.Return",
]


EXPECTED_FEATURE_COUNT = 31


# ==========================================================
# EXPECTED KEYS
# ==========================================================

EXPECTED_KEYS = [
    ".",
    "t",
    "i",
    "e",
    "5",
    "Shift",
    "r",
    "o",
    "a",
    "n",
    "l",
    "Return",
]


# ==========================================================
# NORMALIZE KEY
# ==========================================================

def normalize_key(key: Any) -> str:
    """
    Normalize browser key names.

    Examples:

        R       -> r
        O       -> o
        Enter   -> Return
        Shift   -> Shift
    """

    if key is None:
        return ""

    key = str(key).strip()

    if not key:
        return ""

    if key in ("Enter", "Return"):
        return "Return"

    if key in (
        "Shift",
        "ShiftLeft",
        "ShiftRight",
    ):
        return "Shift"

    if len(key) == 1 and key.isalpha():
        return key.lower()

    return key


# ==========================================================
# EVENT TIMING
# ==========================================================

def get_down(event: Dict[str, Any]) -> Optional[float]:
    """
    Get key-down timestamp.
    """

    value = event.get("key_down")

    if value is None:
        value = event.get("down")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_up(event: Dict[str, Any]) -> Optional[float]:
    """
    Get key-up timestamp.
    """

    value = event.get("key_up")

    if value is None:
        value = event.get("up")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ==========================================================
# EVENT KEY
# ==========================================================

def get_event_key(event: Dict[str, Any]) -> str:

    return normalize_key(
        event.get("key", "")
    )


# ==========================================================
# VALID EVENT
# ==========================================================

def is_valid_event(
    event: Dict[str, Any]
) -> bool:
    """
    Check whether an event has valid timestamps.
    """

    down = get_down(event)
    up = get_up(event)

    if down is None or up is None:
        return False

    if up < down:
        return False

    return True


# ==========================================================
# FIND VALID EVENT
# ==========================================================

def find_event(
    events: List[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, Any]]:
    """
    Find the first valid event for a key.

    Invalid duplicate browser events are skipped.
    """

    wanted = normalize_key(key)

    for event in events:

        if get_event_key(event) != wanted:
            continue

        if is_valid_event(event):
            return event

    return None


# ==========================================================
# FIND ANY EVENT
# ==========================================================

def find_any_event(
    events: List[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, Any]]:
    """
    Find an event even if its timing is invalid.

    Used only for validation/debugging.
    """

    wanted = normalize_key(key)

    for event in events:

        if get_event_key(event) == wanted:
            return event

    return None


# ==========================================================
# HOLD TIME
# ==========================================================

def hold_time(
    events: List[Dict[str, Any]],
    key: str,
) -> float:

    event = find_event(
        events,
        key
    )

    if event is None:

        any_event = find_any_event(
            events,
            key
        )

        if any_event is not None:

            down = get_down(any_event)
            up = get_up(any_event)

            raise ValueError(
                f"Invalid timing for key: {key} "
                f"(down={down}, up={up})"
            )

        raise ValueError(
            f"Missing key: {key}"
        )

    down = get_down(event)
    up = get_up(event)

    return float(up - down)


# ==========================================================
# DOWN-DOWN
# ==========================================================

def down_down(
    events: List[Dict[str, Any]],
    first_key: str,
    second_key: str,
) -> float:

    first = find_event(
        events,
        first_key
    )

    second = find_event(
        events,
        second_key
    )

    if first is None:
        raise ValueError(
            f"Missing key: {first_key}"
        )

    if second is None:
        raise ValueError(
            f"Missing key: {second_key}"
        )

    first_down = get_down(first)
    second_down = get_down(second)

    return float(
        second_down - first_down
    )


# ==========================================================
# UP-DOWN
# ==========================================================

def up_down(
    events: List[Dict[str, Any]],
    first_key: str,
    second_key: str,
) -> float:

    first = find_event(
        events,
        first_key
    )

    second = find_event(
        events,
        second_key
    )

    if first is None:
        raise ValueError(
            f"Missing key: {first_key}"
        )

    if second is None:
        raise ValueError(
            f"Missing key: {second_key}"
        )

    first_up = get_up(first)
    second_down = get_down(second)

    return float(
        second_down - first_up
    )


# ==========================================================
# NORMALIZE EVENTS
# ==========================================================

def normalize_events(
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize browser events and remove malformed entries.

    IMPORTANT:
    We do NOT remove events with negative timing here.
    They are retained so the extractor can choose a
    valid duplicate event when available.
    """

    normalized = []

    for event in events:

        if not isinstance(event, dict):
            continue

        item = dict(event)

        item["key"] = normalize_key(
            event.get("key", "")
        )

        normalized.append(item)

    return normalized


# ==========================================================
# EXTRACT FEATURES
# ==========================================================

def extract_features(
    events: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Convert one password repetition into
    exactly 31 features.

    Returns:

        numpy.ndarray
        shape = (31,)
    """

    if not events:

        raise ValueError(
            "No keystroke events received."
        )

    events = normalize_events(events)

    # ------------------------------------------------------
    # Validate required keys.
    #
    # Only a VALID timing event counts.
    # ------------------------------------------------------

    missing = []

    for key in EXPECTED_KEYS:

        if find_event(
            events,
            key
        ) is None:

            missing.append(key)

    if missing:

        # Give a useful message for Shift.
        if "Shift" in missing:

            raw_shift = find_any_event(
                events,
                "Shift"
            )

            if raw_shift is not None:

                down = get_down(raw_shift)
                up = get_up(raw_shift)

                raise ValueError(
                    "Shift key was captured, but its "
                    "timing is invalid. "
                    f"down={down}, up={up}. "
                    "Please type the uppercase R normally "
                    "using Shift + R."
                )

        raise ValueError(
            "Incomplete password repetition. "
            "Missing or invalid keys: "
            + ", ".join(missing)
        )

    # ======================================================
    # EXACT 31 FEATURES
    # ======================================================

    features = [

        # 1-3
        hold_time(events, "."),
        down_down(events, ".", "t"),
        up_down(events, ".", "t"),

        # 4-6
        hold_time(events, "t"),
        down_down(events, "t", "i"),
        up_down(events, "t", "i"),

        # 7-9
        hold_time(events, "i"),
        down_down(events, "i", "e"),
        up_down(events, "i", "e"),

        # 10-12
        hold_time(events, "e"),
        down_down(events, "e", "5"),
        up_down(events, "e", "5"),

        # 13-15
        hold_time(events, "5"),
        down_down(events, "5", "Shift"),
        up_down(events, "5", "Shift"),

        # 16
        hold_time(events, "Shift"),

        # 17-18
        down_down(events, "Shift", "r"),
        up_down(events, "Shift", "r"),

        # 19-21
        hold_time(events, "o"),
        down_down(events, "o", "a"),
        up_down(events, "o", "a"),

        # 22-24
        hold_time(events, "a"),
        down_down(events, "a", "n"),
        up_down(events, "a", "n"),

        # 25-27
        hold_time(events, "n"),
        down_down(events, "n", "l"),
        up_down(events, "n", "l"),

        # 28-30
        hold_time(events, "l"),
        down_down(events, "l", "Return"),
        up_down(events, "l", "Return"),

        # 31
        hold_time(events, "Return"),
    ]

    # ======================================================
    # NUMPY ARRAY
    # ======================================================

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # ======================================================
    # FINAL CHECK
    # ======================================================

    if features.shape != (31,):

        raise ValueError(
            "Invalid extracted feature shape. "
            f"Expected (31,), got {features.shape}."
        )

    return features


# ==========================================================
# CLASS INTERFACE
# ==========================================================

class FeatureExtractor:

    FEATURE_NAMES = FEATURE_NAMES

    expected_features = EXPECTED_FEATURE_COUNT

    def extract(
        self,
        events: List[Dict[str, Any]]
    ) -> np.ndarray:

        return extract_features(events)

    def extract_features(
        self,
        events: List[Dict[str, Any]]
    ) -> np.ndarray:

        return extract_features(events)


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("KEYSTROKE FEATURE EXTRACTOR TEST")
    print("=" * 60)

    # ------------------------------------------------------
    # Simulated typing of:
    #
    # .tie5Roanl
    #
    # R is uppercase.
    # ------------------------------------------------------

    keys = [
        ".",
        "t",
        "i",
        "e",
        "5",
        "Shift",
        "R",
        "o",
        "a",
        "n",
        "l",
        "Enter",
    ]

    events = []

    timestamp = 100.0

    for key in keys:

        # Normal key.
        down = timestamp
        up = timestamp + 50.0

        events.append(
            {
                "key": key,
                "key_down": down,
                "key_up": up,
            }
        )

        timestamp += 100.0

    try:

        result = extract_features(
            events
        )

        print(
            f"Feature count : {len(result)}"
        )

        print(
            f"Feature shape : {result.shape}"
        )

        print(
            f"Data type     : {result.dtype}"
        )

        print(
            "Expected shape: (31,)"
        )

        if result.shape == (31,):

            print(
                "TEST RESULT   : PASS"
            )

        else:

            print(
                "TEST RESULT   : FAIL"
            )

    except Exception as exc:

        print(
            "TEST RESULT   : FAIL"
        )

        print(
            f"Error         : {exc}"
        )

    print("=" * 60)