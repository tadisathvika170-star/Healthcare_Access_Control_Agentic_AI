"""
capture.py

Stores and manages raw keystroke events before
feature extraction.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class KeystrokeEvent:
    """
    Represents a single key press event.
    """

    key: str
    key_down: float
    key_up: float

    @property
    def hold_time(self):
        """
        Time the key was held.
        """
        return self.key_up - self.key_down


class KeystrokeCapture:
    """
    Stores all keystroke events for one typing session.
    """

    def __init__(self):

        self.events: List[KeystrokeEvent] = []

    def add_event(self, key, key_down, key_up):
        """
        Add a keystroke event.
        """

        event = KeystrokeEvent(
            key=key,
            key_down=key_down,
            key_up=key_up
        )

        self.events.append(event)

    def clear(self):
        """
        Clear current typing session.
        """

        self.events.clear()

    def get_events(self):
        """
        Return all captured events.
        """

        return self.events

    def total_keys(self):
        """
        Number of captured keys.
        """

        return len(self.events)

    def is_empty(self):
        """
        Check whether any events exist.
        """

        return len(self.events) == 0

    def print_events(self):
        """
        Print all events (for debugging).
        """

        print("=" * 60)
        print("Captured Keystrokes")
        print("=" * 60)

        if self.is_empty():

            print("No keystrokes captured.")

            return

        for i, event in enumerate(self.events, start=1):

            print(
                f"{i:02d}. "
                f"Key='{event.key}' "
                f"Down={event.key_down:.4f} "
                f"Up={event.key_up:.4f} "
                f"Hold={event.hold_time:.4f}"
            )

        print("=" * 60)


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    capture = KeystrokeCapture()

    capture.add_event("a", 0.10, 0.23)
    capture.add_event("b", 0.45, 0.60)
    capture.add_event("c", 0.71, 0.84)

    capture.print_events()

    print(f"\nTotal Keys : {capture.total_keys()}")