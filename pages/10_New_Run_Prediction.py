"""Submit a new run to the FastAPI inference service."""

import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="New Run Prediction", page_icon="🔮", layout="wide")
st.title("New Run Prediction")
st.caption("Upload raw run data for versioned FastAPI model inference")

api_url = st.text_input("FastAPI base URL", os.getenv("BIOPROCESS_API_URL", "http://localhost:8000"))
workflow = st.radio("Workflow", ["Tabular baseline", "Multimodal"])
metadata_file = st.file_uploader("Metadata JSON", type=["json"])
sensor_file = st.file_uploader("Sensor time-series CSV", type=["csv"])
notes_file = None
images = []
if workflow == "Multimodal":
    notes_file = st.file_uploader("Notes JSON (optional)", type=["json"])
    images = st.file_uploader("Microscopy images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if metadata_file:
    st.json(json.load(metadata_file))
    metadata_file.seek(0)
if sensor_file:
    preview = pd.read_csv(sensor_file)
    st.dataframe(preview.head(), hide_index=True, use_container_width=True)
    sensor_file.seek(0)

if st.button("Request predictions", type="primary", disabled=not (metadata_file and sensor_file)):
    endpoint = "tabular" if workflow == "Tabular baseline" else "multimodal"
    files = {
        "sensor_file": (sensor_file.name, sensor_file.getvalue(), "text/csv"),
    }
    if notes_file:
        files["notes_file"] = (notes_file.name, notes_file.getvalue(), "application/json")
    if images:
        files = [(key, value) for key, value in files.items()]
        files.extend(("images", (image.name, image.getvalue(), image.type)) for image in images)
    headers = {}
    token = os.getenv("BIOPROCESS_API_TOKEN")
    if token:
        headers["X-API-Key"] = token
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/predict/{endpoint}",
            data={"metadata": metadata_file.getvalue().decode("utf-8")},
            files=files,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        st.success(f"Model version: {result['model_version']}")
        st.dataframe(result["predictions"], hide_index=True, use_container_width=True)
        st.write("Modalities supplied:", ", ".join(result["modalities_present"]))
        for warning in result["warnings"]:
            st.warning(warning)
    except requests.RequestException as error:
        st.error(f"Inference service request failed: {error}")
