"""Streamlit entry point for the Bioprocess Multimodal Intelligence Platform."""

from __future__ import annotations

import sys

import streamlit as st

st.set_page_config(
    page_title="Bioprocess Multimodal Intelligence Platform",
    page_icon="🧬",
    layout="wide",
)

st.title("Bioprocess Multimodal Intelligence Platform")

st.subheader("Status")
st.success("✓ Streamlit working")
st.success(
    "✓ Python environment working "
    f"(Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})"
)
