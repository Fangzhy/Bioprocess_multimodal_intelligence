"""Compare tabular and multimodal titer models."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "multimodal"
st.set_page_config(page_title="Multimodal Predictive Modeling", page_icon="📊", layout="wide")
st.title("Multimodal Predictive Modeling")
st.caption("Fair ablation study using identical batch splits and model families")


@st.cache_data
def load_results():
    return (
        pd.read_csv(ARTIFACT_DIR / "metrics.csv"),
        pd.read_csv(ARTIFACT_DIR / "test_predictions.csv"),
        json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")),
    )


if not (ARTIFACT_DIR / "manifest.json").exists():
    st.error("Multimodal artifacts are missing. Run `python -m src.models.train_multimodal`.")
    st.stop()

metrics, predictions, manifest = load_results()
best = metrics.sort_values("test_rmse").iloc[0]
columns = st.columns(4)
columns[0].metric("Experiments", len(metrics))
columns[1].metric("Best feature set", best["feature_set"])
columns[2].metric("Best model", best["model"])
columns[3].metric("Best test RMSE", f"{best['test_rmse']:.3f} g/L")

st.subheader("Ablation results")
st.dataframe(
    metrics[
        [
            "feature_set", "model", "feature_count", "cv_rmse_mean", "test_rmse",
            "test_mae", "test_r2", "low_titer_precision", "low_titer_recall",
            "low_titer_f1", "training_seconds",
        ]
    ].round(3),
    hide_index=True,
    use_container_width=True,
)

model = st.selectbox("Model family", sorted(metrics["model"].unique()))
model_metrics = metrics.loc[metrics["model"].eq(model)]
st.plotly_chart(
    px.bar(
        model_metrics,
        x="feature_set",
        y="test_rmse",
        color="feature_set",
        title=f"{model}: held-out RMSE by feature set",
    ),
    use_container_width=True,
)

feature_set = st.selectbox("Inspect predictions", manifest["feature_sets"])
selected = predictions.loc[
    predictions["model"].eq(model) & predictions["feature_set"].eq(feature_set)
]
left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.scatter(
            selected,
            x="actual_titer_g_l",
            y="predicted_titer_g_l",
            hover_name="batch_id",
            title="Predicted versus actual",
        ),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        px.scatter(
            selected,
            x="predicted_titer_g_l",
            y="residual_g_l",
            hover_name="batch_id",
            title="Held-out residuals",
        ),
        use_container_width=True,
    )

selected_metric = model_metrics.loc[model_metrics["feature_set"].eq(feature_set)].iloc[0]
st.subheader(f"Low-titer events below {manifest['low_titer_threshold_g_l']:.3f} g/L")
st.dataframe(
    pd.DataFrame(
        [[selected_metric["true_negative"], selected_metric["false_positive"]],
         [selected_metric["false_negative"], selected_metric["true_positive"]]],
        index=["Actual normal", "Actual low"],
        columns=["Predicted normal", "Predicted low"],
    ),
    use_container_width=True,
)

for limitation in manifest["limitations"]:
    st.warning(limitation)
