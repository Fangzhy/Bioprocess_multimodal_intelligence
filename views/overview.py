"""Overview page for the Bioprocess Multimodal Intelligence Platform."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.data.repository import load_batch_catalog, load_images, load_text_records


@st.cache_data
def load_overview_data():
    return load_batch_catalog(), load_text_records(), load_images()


catalog, text_records, images = load_overview_data()

st.title("Bioprocess Multimodal Intelligence Platform")
st.caption(
    "Synthetic educational bioprocess data · Microscopy-style images are illustrative"
)
st.write(
    "Explore process trajectories, experimental metadata, scientist notes, microscopy "
    "images, and final outcomes across fed-batch culture runs."
)

metric_columns = st.columns(5)
metric_columns[0].metric("Batches", f"{len(catalog):,}")
metric_columns[1].metric("Scientist records", f"{len(text_records):,}")
metric_columns[2].metric("Images", f"{len(images):,}")
metric_columns[3].metric(
    "Mean final titer", f"{catalog['final_titer_g_l'].mean():.2f} g/L"
)
metric_columns[4].metric(
    "Mean final viability", f"{catalog['final_viability_pct'].mean():.1f}%"
)

left_chart, right_chart = st.columns(2)
with left_chart:
    titer_figure = px.histogram(
        catalog,
        x="final_titer_g_l",
        color="media_type",
        nbins=14,
        labels={"final_titer_g_l": "Final titer (g/L)", "media_type": "Media"},
        title="Final titer distribution",
    )
    titer_figure.update_layout(yaxis_title="Batch count", bargap=0.08)
    st.plotly_chart(titer_figure, use_container_width=True)

with right_chart:
    relationship_figure = px.scatter(
        catalog,
        x="max_vcd_million_ml",
        y="final_titer_g_l",
        color="media_type",
        hover_name="batch_id",
        symbol="feed_strategy",
        labels={
            "max_vcd_million_ml": "Maximum VCD (million cells/mL)",
            "final_titer_g_l": "Final titer (g/L)",
            "media_type": "Media",
            "feed_strategy": "Feed strategy",
        },
        title="Peak cell density and final titer",
    )
    st.plotly_chart(relationship_figure, use_container_width=True)

st.subheader("Batch outcome summary")
st.dataframe(
    catalog[
        [
            "batch_id",
            "cell_line",
            "media_type",
            "feed_strategy",
            "final_titer_g_l",
            "final_viability_pct",
            "max_vcd_million_ml",
            "quality_metric_score",
        ]
    ],
    hide_index=True,
    use_container_width=True,
)

with st.sidebar:
    st.success("Streamlit working")
    st.info("SQLite data loaded")
