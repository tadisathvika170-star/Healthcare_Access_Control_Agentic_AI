import base64
from pathlib import Path
import streamlit as st


class Background:

    @staticmethod
    def apply(image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            st.warning(f"Background image not found: {image_path}")
            return

        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        page_bg = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """

        st.markdown(page_bg, unsafe_allow_html=True)