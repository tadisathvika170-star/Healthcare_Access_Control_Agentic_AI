"""
component.py

Streamlit component for receiving keystroke data
from the frontend.
"""

import streamlit.components.v1 as components
from pathlib import Path


class KeystrokeComponent:
    """
    Loads the HTML/JS keystroke capture component.
    """

    def __init__(self):

        self.component_path = (
            Path(__file__).parent / "index.html"
        )

    def render(self):
        """
        Display the keystroke capture component.

        Returns
        -------
        dict or None
            Keystroke data returned from JavaScript.
        """

        with open(
            self.component_path,
            "r",
            encoding="utf-8"
        ) as file:

            html = file.read()

        return components.html(
            html,
            height=350,
            scrolling=False,
        )


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    print(
        "This module is intended to run inside Streamlit."
    )