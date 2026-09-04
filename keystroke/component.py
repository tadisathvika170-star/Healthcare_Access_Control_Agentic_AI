from pathlib import Path

import streamlit.components.v1 as components


# Absolute path to the folder containing:
#   index.html
#   script.js
#   style.css (if present)
_COMPONENT_DIR = Path(__file__).resolve().parent

healthcare_keystroke_capture = components.declare_component(
    "healthcare_keystroke_capture",
    path=str(_COMPONENT_DIR),
)


class KeystrokeComponent:
    """
    Bidirectional Streamlit component for capturing keystroke events.

    The frontend sends:
        {
            "submitted": True,
            "submission_id": ...,
            "events": [...]
        }

    Python receives that object as the return value.
    """

    def render(self, key=None):
        return healthcare_keystroke_capture(
            key=key,
            default=None,
        )


# Convenience function
def render_keystroke_component(key=None):
    component = KeystrokeComponent()
    return component.render(key=key)


if __name__ == "__main__":
    print("=" * 60)
    print("KEYSTROKE COMPONENT TEST")
    print("=" * 60)
    print(f"Component directory: {_COMPONENT_DIR}")
    print(f"Directory exists   : {_COMPONENT_DIR.exists()}")
    print(f"index.html exists  : {(_COMPONENT_DIR / 'index.html').exists()}")
    print(f"script.js exists   : {(_COMPONENT_DIR / 'script.js').exists()}")
    print("=" * 60)