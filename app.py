"""Streamlit entry point and explicit application navigation."""

import streamlit as st

st.set_page_config(
    page_title="Bioprocess Multimodal Intelligence Platform",
    page_icon="🧬",
    layout="wide",
)

pages = [
    st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/1_Batch_Explorer.py", title="Batch Explorer", icon="📈"),
    st.Page("pages/2_Batch_Comparison.py", title="Batch Comparison", icon="🔬"),
    st.Page("pages/3_Data_Quality.py", title="Data Quality", icon="✅"),
    st.Page(
        "pages/4_Predictive_Modeling.py",
        title="Predictive Modeling",
        icon="🤖",
    ),
    st.Page("pages/5_Text_Intelligence.py", title="Text Intelligence", icon="📝"),
    st.Page(
        "pages/6_Microscopy_Intelligence.py",
        title="Microscopy Intelligence",
        icon="🖼️",
    ),
    st.Page(
        "pages/7_Multimodal_Investigation.py",
        title="Multimodal Investigation",
        icon="🧩",
    ),
    st.Page(
        "pages/8_Multimodal_Analytics.py",
        title="Multimodal Analytics",
        icon="🔗",
    ),
    st.Page(
        "pages/9_Multimodal_Predictive_Modeling.py",
        title="Multimodal Predictive Modeling",
        icon="📊",
    ),
    st.Page(
        "pages/10_New_Run_Prediction.py",
        title="New Run Prediction",
        icon="🔮",
    ),
    st.Page(
        "pages/11_Scientific_Copilot.py",
        title="Scientific Copilot",
        icon="💬",
    ),
]

navigation = st.navigation(pages)
navigation.run()
