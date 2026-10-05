"""Explain an interpretable evidence-level multimodal risk score."""

import plotly.express as px
import streamlit as st

from src.models.multimodal import build_multimodal_feature_table, calculate_risk_scores

st.set_page_config(page_title="Multimodal Analytics", page_icon="🧩", layout="wide")
st.title("Multimodal Analytics")
st.caption("Interpretable late fusion · Heuristic investigation score")


@st.cache_data
def load_scores():
    return calculate_risk_scores(build_multimodal_feature_table())


scores = load_scores()
batch_ids = sorted(scores["batch_id"])
selected_id = st.selectbox("Batch", batch_ids, index=batch_ids.index("B014"))
selected = scores.loc[scores["batch_id"].eq(selected_id)].iloc[0]

columns = st.columns(4)
columns[0].metric("Sensor anomaly", f"{selected['sensor_anomaly']:.2f}")
columns[1].metric("Text anomaly", f"{selected['text_anomaly']:.2f}")
columns[2].metric("Image anomaly", f"{selected['image_anomaly']:.2f}")
columns[3].metric("Combined risk", f"{selected['multimodal_risk_score']:.2f}")

component_data = {
    "component": ["Sensor", "Text", "Image"],
    "score": [selected["sensor_anomaly"], selected["text_anomaly"], selected["image_anomaly"]],
    "weight": [0.5, 0.2, 0.3],
}
st.plotly_chart(
    px.bar(component_data, x="component", y="score", color="weight", range_y=[0, 1]),
    use_container_width=True,
)
st.subheader("All batches")
st.dataframe(scores.round(3), hide_index=True, use_container_width=True)
st.info(
    "The score is a weighted combination of percentile-ranked evidence: 0.5 sensor, "
    "0.2 text, and 0.3 image. It is an investigation aid, not a calibrated probability. "
    "If a modality is missing, available weights are renormalized."
)
