"""Plotly figures shared by Streamlit dashboard pages."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

VARIABLE_LABELS = {
    "temperature_c": "Temperature (°C)",
    "ph": "pH",
    "do_pct": "DO (% air saturation)",
    "agitation_rpm": "Agitation (rpm)",
    "air_flow_slpm": "Air flow (SLPM)",
    "o2_flow_slpm": "O₂ flow (SLPM)",
    "co2_flow_slpm": "CO₂ flow (SLPM)",
    "feed_rate_ml_hr": "Feed rate (mL/h)",
    "glucose_g_l": "Glucose (g/L)",
    "lactate_g_l": "Lactate (g/L)",
    "viable_cell_density_million_ml": "VCD (million cells/mL)",
    "viability_pct": "Viability (%)",
    "titer_g_l": "Titer (g/L)",
}


def make_trajectory_figure(
    sensor_data: pd.DataFrame,
    variables: Sequence[str],
    *,
    title: str,
) -> go.Figure:
    if not variables:
        return go.Figure()
    figure = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.3 / len(variables)),
    )
    batch_id = str(sensor_data["batch_id"].iloc[0])
    for row_number, variable in enumerate(variables, start=1):
        figure.add_trace(
            go.Scatter(
                x=sensor_data["time_hr"],
                y=sensor_data[variable],
                name=batch_id,
                mode="lines",
                line={"width": 2},
                showlegend=row_number == 1,
            ),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(title_text=VARIABLE_LABELS[variable], row=row_number, col=1)
    figure.update_xaxes(title_text="Time (h)", row=len(variables), col=1)
    figure.update_layout(
        title=title,
        height=max(320, 220 * len(variables)),
        margin={"l": 80, "r": 30, "t": 70, "b": 50},
        hovermode="x unified",
    )
    return figure


def make_reference_comparison_figure(
    selected_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    variables: Sequence[str],
    *,
    selected_batch_id: str,
) -> go.Figure:
    if not variables:
        return go.Figure()
    figure = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.3 / len(variables)),
    )
    for row_number, variable in enumerate(variables, start=1):
        reference = reference_data.groupby("time_hr")[variable].agg(["mean", "std"])
        upper = reference["mean"] + reference["std"].fillna(0)
        lower = reference["mean"] - reference["std"].fillna(0)
        show_legend = row_number == 1
        figure.add_trace(
            go.Scatter(
                x=reference.index,
                y=upper,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=reference.index,
                y=lower,
                name="Reference ±1 SD",
                fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.16)",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=show_legend,
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=reference.index,
                y=reference["mean"],
                name="Reference mean",
                line={"color": "#636EFA", "dash": "dash"},
                showlegend=show_legend,
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=selected_data["time_hr"],
                y=selected_data[variable],
                name=selected_batch_id,
                line={"color": "#EF553B", "width": 2.5},
                showlegend=show_legend,
            ),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(title_text=VARIABLE_LABELS[variable], row=row_number, col=1)
    figure.update_xaxes(title_text="Time (h)", row=len(variables), col=1)
    figure.update_layout(
        title=f"{selected_batch_id} versus reference cohort",
        height=max(350, 240 * len(variables)),
        margin={"l": 80, "r": 30, "t": 70, "b": 50},
        hovermode="x unified",
    )
    return figure
