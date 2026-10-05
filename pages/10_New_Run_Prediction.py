"""Submit uploaded or built-in example runs to the FastAPI inference service."""

from __future__ import annotations

import io
import json
import os

import pandas as pd
import requests
import streamlit as st

from src.inference.examples import load_example_run
from src.inference.service import predict_request

st.set_page_config(page_title="New Run Prediction", page_icon="🔮", layout="wide")
st.title("New Run Prediction")
st.caption("Inspect, edit, and submit raw run data to versioned model pipelines")


@st.cache_data
def get_example_run():
    return load_example_run()


example = get_example_run()
st.session_state.setdefault("prediction_input_source", "Upload files")
st.session_state.setdefault("prediction_workflow", "Tabular baseline")


def select_example(workflow: str) -> None:
    st.session_state["prediction_input_source"] = "Built-in example"
    st.session_state["prediction_workflow"] = workflow


st.subheader("Try a built-in example")
st.write(
    "Load a complete synthetic request, inspect every field, change values, and send it "
    "through the same API used for uploaded files."
)
button_columns = st.columns(2)
button_columns[0].button(
    "Load tabular example",
    on_click=select_example,
    args=("Tabular baseline",),
    width="stretch",
)
button_columns[1].button(
    "Load multimodal example",
    on_click=select_example,
    args=("Multimodal",),
    width="stretch",
)

prediction_engine = st.radio(
    "Prediction engine",
    ["Built-in demo", "FastAPI service"],
    horizontal=True,
    help=(
        "Built-in demo works on Streamlit Community Cloud. FastAPI requires a separately "
        "deployed backend reachable over HTTPS."
    ),
)
api_url = os.getenv("BIOPROCESS_API_URL", "http://localhost:8000")
if prediction_engine == "FastAPI service":
    api_url = st.text_input("FastAPI base URL", api_url)
    if "localhost" in api_url or "127.0.0.1" in api_url:
        st.warning(
            "A localhost URL works only when FastAPI runs on the same machine. On Streamlit "
            "Community Cloud, configure the public HTTPS URL of a separately deployed API."
        )
else:
    st.info(
        "Predictions run inside this Streamlit app with the same saved pipelines used by "
        "FastAPI. This mode is suitable for the hosted portfolio demo."
    )
input_source = st.radio(
    "Input source",
    ["Upload files", "Built-in example"],
    key="prediction_input_source",
    horizontal=True,
)
workflow = st.radio(
    "Workflow",
    ["Tabular baseline", "Multimodal"],
    key="prediction_workflow",
    horizontal=True,
)

metadata_content: bytes | None = None
sensor_content: bytes | None = None
notes_content: bytes | None = None
image_payloads: list[tuple[str, bytes, str]] = []
example_text_embedding = None
example_image_embedding = None

if input_source == "Built-in example":
    st.info(
        "This editable example is derived from synthetic batch B018 and relabeled "
        "NEW_DEMO_001. Titer and outcomes are excluded from the request."
    )
    metadata_text = st.text_area(
        "Metadata JSON",
        value=json.dumps(example.metadata, indent=2),
        height=235,
        help="Edit categories or numeric settings before requesting a prediction.",
    )
    try:
        parsed_metadata = json.loads(metadata_text)
        st.json(parsed_metadata)
        metadata_content = metadata_text.encode("utf-8")
    except json.JSONDecodeError as error:
        st.error(f"Metadata JSON is invalid: {error}")

    st.markdown("**Sensor time-series CSV**")
    edited_sensor = st.data_editor(
        example.sensor_data,
        key="example_sensor_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        height=330,
    )
    sensor_content = edited_sensor.to_csv(index=False).encode("utf-8")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download metadata JSON",
        data=metadata_text,
        file_name="example_metadata.json",
        mime="application/json",
        width="stretch",
    )
    download_columns[1].download_button(
        "Download sensor CSV",
        data=sensor_content,
        file_name="example_sensor_data.csv",
        mime="text/csv",
        width="stretch",
    )

    if workflow == "Multimodal":
        st.markdown("**Timestamped notes JSON**")
        edited_notes = st.data_editor(
            example.notes,
            key="example_notes_editor",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
        )
        notes_content = edited_notes.to_json(orient="records", indent=2).encode("utf-8")
        if edited_notes.equals(example.notes):
            example_text_embedding = example.text_embedding
        st.download_button(
            "Download notes JSON",
            data=notes_content,
            file_name="example_notes.json",
            mime="application/json",
        )

        st.markdown("**Synthetic microscopy images**")
        st.caption(
            "The unchanged built-in notes and images use committed example embeddings, so "
            "this demonstration does not need to download encoder checkpoints. Editing the "
            "notes triggers live Sentence Transformer encoding."
        )
        image_columns = st.columns(len(example.image_paths))
        for column, path in zip(image_columns, example.image_paths, strict=True):
            with column:
                st.image(str(path), width="stretch", caption=path.name)
            image_payloads.append((path.name, path.read_bytes(), "image/png"))
        example_image_embedding = example.image_embedding
else:
    metadata_file = st.file_uploader("Metadata JSON", type=["json"])
    sensor_file = st.file_uploader("Sensor time-series CSV", type=["csv"])
    notes_file = None
    uploaded_images = []
    if workflow == "Multimodal":
        notes_file = st.file_uploader("Notes JSON (optional)", type=["json"])
        uploaded_images = st.file_uploader(
            "Microscopy images (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    if metadata_file:
        metadata_content = metadata_file.getvalue()
        try:
            st.json(json.loads(metadata_content))
        except json.JSONDecodeError as error:
            st.error(f"Metadata JSON is invalid: {error}")
            metadata_content = None
    if sensor_file:
        sensor_content = sensor_file.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(sensor_content))
            st.dataframe(preview.head(10), hide_index=True, width="stretch")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            st.error(f"Sensor CSV cannot be read: {error}")
            sensor_content = None
    if notes_file:
        notes_content = notes_file.getvalue()
    image_payloads = [
        (image.name, image.getvalue(), image.type) for image in uploaded_images
    ]

ready = metadata_content is not None and sensor_content is not None
if st.button("Request predictions", type="primary", disabled=not ready):
    endpoint = "tabular" if workflow == "Tabular baseline" else "multimodal"
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("sensor_file", ("sensor_data.csv", sensor_content, "text/csv"))
    ]
    if workflow == "Multimodal" and notes_content:
        files.append(
            ("notes_file", ("notes.json", notes_content, "application/json"))
        )
    if workflow == "Multimodal":
        files.extend(("images", payload) for payload in image_payloads)

    try:
        if prediction_engine == "Built-in demo":
            result = predict_request(
                endpoint,
                metadata_content,
                sensor_content,
                notes_content if workflow == "Multimodal" else None,
                [content for _, content, _ in image_payloads]
                if workflow == "Multimodal"
                else None,
                example_text_embedding,
                example_image_embedding,
            ).model_dump()
        else:
            headers = {}
            token = os.getenv("BIOPROCESS_API_TOKEN")
            if token:
                headers["X-API-Key"] = token
            response = requests.post(
                f"{api_url.rstrip('/')}/v1/predict/{endpoint}",
                data={"metadata": metadata_content.decode("utf-8")},
                files=files,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        st.success(f"Model version: {result['model_version']}")
        st.dataframe(result["predictions"], hide_index=True, width="stretch")
        st.write("Modalities supplied:", ", ".join(result["modalities_present"]))
        for warning in result["warnings"]:
            st.warning(warning)
    except requests.RequestException as error:
        st.error(f"FastAPI inference request failed: {error}")
    except (ValueError, OSError) as error:
        st.error(f"Built-in inference failed: {error}")
