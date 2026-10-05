"""Tests for Milestone 4 dashboard data and charts."""

from __future__ import annotations

import pytest

from src.data.quality import build_quality_report
from src.data.repository import (
    load_batch_catalog,
    load_sensor_data,
    load_table,
    select_reference_batches,
)
from src.visualization.charts import (
    make_reference_comparison_figure,
    make_trajectory_figure,
)


def test_repository_queries_and_reference_cohort() -> None:
    catalog = load_batch_catalog()
    assert len(catalog) == 50

    b014 = load_sensor_data(["B014"])
    assert len(b014) == 61
    assert b014["time_hr"].tolist() == list(range(0, 241, 4))

    references = select_reference_batches(catalog, "B014", "Same media")
    assert "B014" not in references
    selected_media = catalog.loc[catalog["batch_id"].eq("B014"), "media_type"].iloc[0]
    reference_media = catalog.loc[
        catalog["batch_id"].isin(references), "media_type"
    ]
    assert not reference_media.empty
    assert reference_media.eq(selected_media).all()


def test_repository_rejects_unknown_inputs() -> None:
    catalog = load_batch_catalog()
    with pytest.raises(ValueError, match="Unsupported table"):
        load_table("not_a_table")
    with pytest.raises(ValueError, match="Unknown batch"):
        select_reference_batches(catalog, "B999", "Same media")
    with pytest.raises(ValueError, match="Unsupported cohort"):
        select_reference_batches(catalog, "B014", "Unknown cohort")


def test_quality_report_matches_complete_synthetic_dataset() -> None:
    tables = {
        table_name: load_table(table_name)
        for table_name in (
            "batches",
            "sensor_data",
            "outcomes",
            "text_records",
            "images",
        )
    }
    report = build_quality_report(tables)

    assert report.total_missing == 0
    assert report.total_duplicates == 0
    assert report.total_range_violations == 0
    assert report.total_outliers > 0
    assert len(report.batch_coverage) == 50
    assert report.batch_coverage["measurement_count"].eq(61).all()


def test_dashboard_charts_have_expected_traces() -> None:
    measurements = load_sensor_data(["B014", "B015"])
    selected = measurements.loc[measurements["batch_id"].eq("B014")]
    reference = measurements.loc[measurements["batch_id"].eq("B015")]

    trajectory = make_trajectory_figure(selected, ["ph", "do_pct"], title="Test")
    comparison = make_reference_comparison_figure(
        selected,
        reference,
        ["ph", "do_pct"],
        selected_batch_id="B014",
    )
    assert len(trajectory.data) == 2
    assert len(comparison.data) == 8
