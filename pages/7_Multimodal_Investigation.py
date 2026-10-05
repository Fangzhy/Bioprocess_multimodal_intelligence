"""Align evidence from multiple modalities within an event window."""

from pathlib import Path

import streamlit as st

from src.data.events import align_event
from src.data.repository import load_batch_catalog
from src.visualization.charts import make_trajectory_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Multimodal Investigation", page_icon="🧩", layout="wide")
st.title("Multimodal Investigation")
st.caption("Time-aligned sensor, note, image, and retrospective outcome evidence")

catalog = load_batch_catalog()
batch_ids = catalog["batch_id"].tolist()
controls = st.columns(3)
batch_id = controls[0].selectbox("Batch", batch_ids, index=batch_ids.index("B014"))
start_hr = controls[1].number_input("Start hour", 0, 240, 72, step=4)
end_hr = controls[2].number_input("End hour", 0, 240, 96, step=4)

if end_hr < start_hr:
    st.error("End hour must be at or after start hour.")
    st.stop()

event = align_event(batch_id, int(start_hr), int(end_hr))
st.plotly_chart(
    make_trajectory_figure(
        event.sensors,
        ["ph", "do_pct", "lactate_g_l", "viable_cell_density_million_ml"],
        title=f"{batch_id}: {start_hr}-{end_hr} h",
    ),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Sensor summary")
    st.dataframe(event.sensor_summary, hide_index=True, use_container_width=True)
    st.subheader("Scientist notes")
    if event.notes.empty:
        st.info("No notes fall inside this event window.")
    else:
        st.dataframe(event.notes, hide_index=True, use_container_width=True)
with right:
    st.subheader("Microscopy")
    if event.images.empty:
        st.info("No image capture falls inside this event window.")
    else:
        for _, image in event.images.iterrows():
            st.image(
                str(PROJECT_ROOT / image["file_path"]),
                width=260,
                caption=f"{image['time_hr']} h · synthetic illustration",
            )
    st.subheader("Final outcome (retrospective context)")
    st.dataframe(
        event.outcome[["final_titer_g_l", "final_viability_pct", "max_vcd_million_ml"]],
        hide_index=True,
        use_container_width=True,
    )

st.info("Evidence is selected inclusively when its timestamp falls between the start and end hours.")
