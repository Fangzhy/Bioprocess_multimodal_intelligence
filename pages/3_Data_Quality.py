"""Inspect missingness, duplicates, ranges, outliers, and batch coverage."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.data.quality import build_quality_report
from src.data.repository import TABLE_NAMES, load_table

st.set_page_config(page_title="Data Quality", page_icon="✅", layout="wide")


@st.cache_data
def load_all_tables():
    return {table_name: load_table(table_name) for table_name in TABLE_NAMES}


tables = load_all_tables()
report = build_quality_report(tables)

st.title("Data Quality")
st.caption(
    "Synthetic educational data · Statistical outliers can represent meaningful "
    "process events and are not automatically data errors"
)

metric_columns = st.columns(4)
metric_columns[0].metric("Missing values", f"{report.total_missing:,}")
metric_columns[1].metric("Duplicate keys", f"{report.total_duplicates:,}")
metric_columns[2].metric("Range violations", f"{report.total_range_violations:,}")
metric_columns[3].metric("IQR outlier flags", f"{report.total_outliers:,}")

st.subheader("Missing values")
missing = report.missing_values.loc[report.missing_values["missing_count"].gt(0)]
if missing.empty:
    st.success("No missing values were found in the five database tables.")
else:
    st.dataframe(missing, hide_index=True, use_container_width=True)

st.subheader("Duplicate keys")
st.dataframe(report.duplicates, hide_index=True, use_container_width=True)

st.subheader("Expected-range checks")
st.dataframe(report.range_violations, hide_index=True, use_container_width=True)

st.subheader("Statistical outlier screen")
st.write(
    "Outliers use the global 1.5 × IQR rule. Review them as investigation candidates; "
    "do not remove them automatically."
)
st.dataframe(report.outliers, hide_index=True, use_container_width=True)

st.subheader("Measurement coverage by batch")
coverage_figure = px.bar(
    report.batch_coverage,
    x="batch_id",
    y="measurement_count",
    labels={"batch_id": "Batch", "measurement_count": "Measurements"},
)
coverage_figure.update_layout(xaxis={"categoryorder": "category ascending"})
st.plotly_chart(coverage_figure, use_container_width=True)
st.dataframe(report.batch_coverage, hide_index=True, use_container_width=True)
