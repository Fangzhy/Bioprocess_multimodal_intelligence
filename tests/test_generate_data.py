from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.generate_data import SCENARIOS, SimulationConfig, write_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_dataset_shape_and_planted_scenarios(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path)

    assert len(dataset.batches) == 50
    assert len(dataset.sensor_data) == 3_050
    assert len(dataset.images) == 150
    assert dataset.sensor_data.groupby("batch_id").size().eq(61).all()
    assert dataset.sensor_data.groupby("batch_id")["time_hr"].apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert set(dataset.scenario_truth.query("is_abnormal")["batch_id"]) == set(
        SCENARIOS
    )
    assert dataset.images["is_synthetic"].all()
    assert all((tmp_path / path).is_file() for path in dataset.images["file_path"])
    assert dataset.batches["batch_id"].is_unique
    assert dataset.outcomes["batch_id"].is_unique
    assert set(dataset.text_records["batch_id"]) <= set(dataset.batches["batch_id"])
    assert set(dataset.images["batch_id"]) <= set(dataset.batches["batch_id"])
    assert dataset.sensor_data["ph"].between(6.75, 7.25).all()
    assert dataset.sensor_data["viability_pct"].between(70, 100).all()
    assert dataset.sensor_data.groupby("batch_id")["titer_g_l"].apply(
        lambda values: values.is_monotonic_increasing
    ).all()

    b014 = dataset.sensor_data.query("batch_id == 'B014' and 72 <= time_hr <= 96")
    normal = dataset.sensor_data.query("batch_id == 'B001' and 72 <= time_hr <= 96")
    assert b014["do_pct"].std() > 2 * normal["do_pct"].std()

    b007_lactate = dataset.sensor_data.query("batch_id == 'B007' and time_hr == 120")[
        "lactate_g_l"
    ].iloc[0]
    normal_lactate = dataset.sensor_data.query(
        "batch_id == 'B001' and time_hr == 120"
    )["lactate_g_l"].iloc[0]
    assert b007_lactate > normal_lactate + 1

    normal_ids = [
        batch_id
        for batch_id in dataset.batches["batch_id"]
        if batch_id not in SCENARIOS
    ]
    outcomes = dataset.outcomes.set_index("batch_id")
    assert outcomes.loc["B023", "max_vcd_million_ml"] < outcomes.loc[
        normal_ids, "max_vcd_million_ml"
    ].median()
    assert outcomes.loc["B044", "final_titer_g_l"] < outcomes.loc[
        normal_ids, "final_titer_g_l"
    ].median()


def test_generation_is_reproducible(tmp_path: Path) -> None:
    config = SimulationConfig(n_batches=3, image_size_px=64)
    first = write_dataset(tmp_path / "first", config)
    second = write_dataset(tmp_path / "second", config)

    pd.testing.assert_frame_equal(first.batches, second.batches)
    pd.testing.assert_frame_equal(first.sensor_data, second.sensor_data)
    pd.testing.assert_frame_equal(first.outcomes, second.outcomes)
    pd.testing.assert_frame_equal(first.text_records, second.text_records)
    pd.testing.assert_frame_equal(first.images, second.images)

    first_image = tmp_path / "first" / first.images.iloc[0]["file_path"]
    second_image = tmp_path / "second" / second.images.iloc[0]["file_path"]
    assert _sha256(first_image) == _sha256(second_image)
