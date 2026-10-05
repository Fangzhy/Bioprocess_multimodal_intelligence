"""Explore one batch's measurements, metadata, notes, and images."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.repository import (
    PROJECT_ROOT,
    load_batch_catalog,
    load_images,
    load_sensor_data,
    load_text_records,
)
from src.visualization.charts import VARIABLE_LABELS, make_trajectory_figure

st.set_page_config(page_title="Batch Explorer", page_icon="📈", layout="wide")


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return load_batch_catalog()


@st.cache_data
def load_batch(batch_id: str):
    return (
        load_sensor_data([batch_id]),
        load_text_records(batch_id),
        load_images(batch_id),
    )


catalog = load_catalog()
batch_ids = catalog["batch_id"].tolist()
default_index = batch_ids.index("B014") if "B014" in batch_ids else 0

st.title("Batch Explorer")
st.caption(
    "Synthetic educational bioprocess data · Microscopy-style images are illustrative"
)

selected_batch = st.selectbox("Choose batch", batch_ids, index=default_index)
default_variables = [
    "ph",
    "do_pct",
    "viable_cell_density_million_ml",
    "lactate_g_l",
    "titer_g_l",
]
selected_variables = st.multiselect(
    "Variables",
    options=list(VARIABLE_LABELS),
    default=default_variables,
    format_func=lambda value: VARIABLE_LABELS[value],
)

sensor_data, notes, image_records = load_batch(selected_batch)
metadata = catalog.set_index("batch_id").loc[selected_batch]

summary_columns = st.columns(5)
summary_columns[0].metric("Cell line", metadata["cell_line"])
summary_columns[1].metric("Media", metadata["media_type"])
summary_columns[2].metric("Feed", metadata["feed_strategy"])
summary_columns[3].metric("Final titer", f"{metadata['final_titer_g_l']:.2f} g/L")
summary_columns[4].metric(
    "Final viability", f"{metadata['final_viability_pct']:.1f}%"
)

if selected_variables:
    st.plotly_chart(
        make_trajectory_figure(
            sensor_data,
            selected_variables,
            title=f"{selected_batch} process trajectories",
        ),
        use_container_width=True,
    )
    summary = (
        sensor_data[selected_variables]
        .agg(["mean", "std", "min", "max"])
        .transpose()
        .rename_axis("variable")
        .reset_index()
    )
    summary.insert(1, "label", summary["variable"].map(VARIABLE_LABELS))
    summary["missing"] = sensor_data[selected_variables].isna().sum().to_numpy()
    st.subheader("Selected-variable statistics")
    st.dataframe(summary, hide_index=True, use_container_width=True)
else:
    st.info("Select at least one variable to display process trajectories.")

st.subheader("Scientist notes and observations")
st.dataframe(
    notes[["time_hr", "text_type", "content"]],
    hide_index=True,
    use_container_width=True,
)

st.subheader("Illustrative microscopy images")
image_columns = st.columns(max(1, len(image_records)))
for column, record in zip(image_columns, image_records.itertuples(), strict=True):
    with column:
        st.image(
            str(PROJECT_ROOT / record.file_path),
            caption=f"{record.time_hr} h · synthetic",
            use_container_width=True,
        )
