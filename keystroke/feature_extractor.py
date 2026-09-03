"""
feature_extractor.py

Extracts numerical features from captured keystroke events.
"""

import numpy as np


class FeatureExtractor:
    """
    Extract features from keystroke events.
    Produces a fixed-length feature vector (33 features).
    """

    def __init__(self):
        self.expected_features = 33

    def extract(self, events):
        """
        Parameters
        ----------
        events : list
            List of KeystrokeEvent objects.

        Returns
        -------
        numpy.ndarray
            Feature vector of length 33.
        """

        if len(events) < 2:
            raise ValueError(
                "At least 2 keystroke events are required."
            )

        hold_times = []
        flight_times = []

        # Hold time for each key
        for event in events:
            hold_times.append(event.hold_time)

        # Flight time between consecutive keys
        for i in range(len(events) - 1):
            flight = (
                events[i + 1].key_down
                - events[i].key_up
            )
            flight_times.append(flight)

        features = []

        # Hold statistics
        features.extend([
            np.mean(hold_times),
            np.std(hold_times),
            np.min(hold_times),
            np.max(hold_times),
        ])

        # Flight statistics
        features.extend([
            np.mean(flight_times),
            np.std(flight_times),
            np.min(flight_times),
            np.max(flight_times),
        ])

        # Raw hold times
        features.extend(hold_times)

        # Raw flight times
        features.extend(flight_times)

        # Ensure fixed length
        if len(features) < self.expected_features:
            features.extend(
                [0.0] * (self.expected_features - len(features))
            )

        # Trim if longer
        features = features[:self.expected_features]

        return np.array(features, dtype=np.float32)


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    from keystroke.capture import (
        KeystrokeCapture
    )

    capture = KeystrokeCapture()

    capture.add_event("a", 0.10, 0.22)
    capture.add_event("b", 0.35, 0.49)
    capture.add_event("c", 0.62, 0.76)
    capture.add_event("d", 0.91, 1.05)

    extractor = FeatureExtractor()

    features = extractor.extract(
        capture.get_events()
    )

    print("=" * 60)
    print("Feature Extraction Test")
    print("=" * 60)

    print("Feature Vector Length :", len(features))
    print("\nFeatures:\n")
    print(features)