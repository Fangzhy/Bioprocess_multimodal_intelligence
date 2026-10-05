"""Compare one batch with a transparent historical reference cohort."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.repository import (
    load_batch_catalog,
    load_sensor_data,
    select_reference_batches,
)
from src.visualization.charts import (
    VARIABLE_LABELS,
    make_reference_comparison_figure,
)

st.set_page_config(page_title="Batch Comparison", page_icon="🔬", layout="wide")


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return load_batch_catalog()


@st.cache_data
def load_measurements(batch_ids: tuple[str, ...]) -> pd.DataFrame:
    return load_sensor_data(batch_ids)


catalog = load_catalog()
batch_ids = catalog["batch_id"].tolist()
default_index = batch_ids.index("B014") if "B014" in batch_ids else 0

st.title("Batch Comparison")
st.caption(
    "Synthetic educational data · Reference cohorts use metadata and exclude the "
    "selected batch"
)

control_columns = st.columns(2)
with control_columns[0]:
    selected_batch = st.selectbox("Investigated batch", batch_ids, index=default_index)
with control_columns[1]:
    cohort = st.selectbox(
        "Reference cohort",
        ["Same media and cell line", "Same media", "All other batches"],
        index=1,
    )

reference_ids = select_reference_batches(catalog, selected_batch, cohort)
if not reference_ids:
    st.warning("The selected cohort has no reference batches. Choose a broader cohort.")
    st.stop()

st.write(f"Reference cohort: **{cohort}** · **{len(reference_ids)} batches**")
variables = st.multiselect(
    "Comparison variables",
    options=list(VARIABLE_LABELS),
    default=[
        "do_pct",
        "viable_cell_density_million_ml",
        "lactate_g_l",
        "titer_g_l",
    ],
    format_func=lambda value: VARIABLE_LABELS[value],
)

measurements = load_measurements((selected_batch, *reference_ids))
selected_data = measurements.loc[measurements["batch_id"].eq(selected_batch)]
reference_data = measurements.loc[measurements["batch_id"].isin(reference_ids)]

if variables:
    st.plotly_chart(
        make_reference_comparison_figure(
            selected_data,
            reference_data,
            variables,
            selected_batch_id=selected_batch,
        ),
        use_container_width=True,
    )

    comparison_rows = []
    for variable in variables:
        selected_mean = float(selected_data[variable].mean())
        reference_batch_means = reference_data.groupby("batch_id")[variable].mean()
        reference_mean = float(reference_batch_means.mean())
        reference_std = float(reference_batch_means.std())
        comparison_rows.append(
            {
                "variable": VARIABLE_LABELS[variable],
                "selected_mean": selected_mean,
                "reference_mean": reference_mean,
                "reference_batch_std": reference_std,
                "difference": selected_mean - reference_mean,
                "difference_pct": (
                    100 * (selected_mean - reference_mean) / reference_mean
                    if reference_mean
                    else float("nan")
                ),
            }
        )
    st.subheader("Batch-level comparison")
    st.dataframe(
        pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True
    )
else:
    st.info("Select at least one variable to compare.")

st.subheader("Outcome comparison")
selected_outcome = catalog.loc[catalog["batch_id"].eq(selected_batch)]
reference_outcomes = catalog.loc[catalog["batch_id"].isin(reference_ids)]
outcome_summary = pd.DataFrame(
    {
        "metric": ["Final titer (g/L)", "Final viability (%)", "Maximum VCD"],
        selected_batch: [
            selected_outcome["final_titer_g_l"].iloc[0],
            selected_outcome["final_viability_pct"].iloc[0],
            selected_outcome["max_vcd_million_ml"].iloc[0],
        ],
        "reference_mean": [
            reference_outcomes["final_titer_g_l"].mean(),
            reference_outcomes["final_viability_pct"].mean(),
            reference_outcomes["max_vcd_million_ml"].mean(),
        ],
        "reference_std": [
            reference_outcomes["final_titer_g_l"].std(),
            reference_outcomes["final_viability_pct"].std(),
            reference_outcomes["max_vcd_million_ml"].std(),
        ],
    }
)
st.dataframe(outcome_summary, hide_index=True, use_container_width=True)
