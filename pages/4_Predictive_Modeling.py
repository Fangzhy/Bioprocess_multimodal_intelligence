"""Compare tabular models for end-of-run final-titer prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tabular"

st.set_page_config(page_title="Predictive Modeling", page_icon="🤖", layout="wide")
st.title("Predictive Modeling")
st.caption("Tabular end-of-run models · Synthetic educational data")


@st.cache_data
def load_model_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics.csv")
    predictions = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    importance = pd.read_csv(ARTIFACT_DIR / "feature_importance.csv")
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    return metrics, predictions, importance, manifest


if not (ARTIFACT_DIR / "manifest.json").exists():
    st.error("Model artifacts are missing. Run `python -m src.models.train_tabular`.")
    st.stop()

metrics, predictions, importance, manifest = load_model_results()
best = metrics.sort_values("test_rmse").iloc[0]
summary_columns = st.columns(4)
summary_columns[0].metric("Training batches", manifest["data"]["training_count"])
summary_columns[1].metric("Held-out batches", manifest["data"]["test_count"])
summary_columns[2].metric("Best test RMSE", f"{best['test_rmse']:.3f} g/L")
summary_columns[3].metric("Best model", best["model"])

st.subheader("Model comparison")
display_metrics = metrics[
    ["model", "cv_rmse_mean", "cv_rmse_std", "test_rmse", "test_mae", "test_r2"]
].copy()
st.dataframe(display_metrics.round(3), hide_index=True, use_container_width=True)

selected_model = st.selectbox("Inspect model", metrics["model"].tolist())
selected_predictions = predictions.loc[predictions["model"].eq(selected_model)]

left, right = st.columns(2)
with left:
    prediction_figure = px.scatter(
        selected_predictions,
        x="actual_titer_g_l",
        y="predicted_titer_g_l",
        hover_name="batch_id",
        labels={
            "actual_titer_g_l": "Actual final titer (g/L)",
            "predicted_titer_g_l": "Predicted final titer (g/L)",
        },
        title=f"{selected_model}: predicted versus actual",
    )
    limits = [
        min(
            selected_predictions["actual_titer_g_l"].min(),
            selected_predictions["predicted_titer_g_l"].min(),
        ),
        max(
            selected_predictions["actual_titer_g_l"].max(),
            selected_predictions["predicted_titer_g_l"].max(),
        ),
    ]
    prediction_figure.add_shape(
        type="line", x0=limits[0], y0=limits[0], x1=limits[1], y1=limits[1],
        line={"dash": "dash", "color": "gray"},
    )
    st.plotly_chart(prediction_figure, use_container_width=True)

with right:
    residual_figure = px.scatter(
        selected_predictions,
        x="predicted_titer_g_l",
        y="residual_g_l",
        hover_name="batch_id",
        labels={
            "predicted_titer_g_l": "Predicted final titer (g/L)",
            "residual_g_l": "Residual: actual - predicted (g/L)",
        },
        title=f"{selected_model}: held-out residuals",
    )
    residual_figure.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(residual_figure, use_container_width=True)

st.subheader("Feature importance")
selected_importance = importance.loc[importance["model"].eq(selected_model)].head(12)
if selected_importance.empty:
    st.info("This model does not expose compatible feature importance values.")
else:
    importance_figure = px.bar(
        selected_importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"{selected_model}: strongest fitted feature signals",
    )
    st.plotly_chart(importance_figure, use_container_width=True)

with st.expander("Experiment design and limitations"):
    st.write(
        "Each row represents one batch. The fixed held-out set contains complete batches, "
        "and five-fold cross-validation runs only within the training batches. Preprocessing "
        "is fitted separately inside each fold."
    )
    st.write(
        "These models use process measurements through 240 hours. They describe an end-of-run "
        "prediction experiment and are not early-process forecasts."
    )
    for limitation in manifest["limitations"]:
        st.markdown(f"- {limitation}")
