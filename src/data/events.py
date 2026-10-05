"""Align sensor, note, image, and outcome evidence to an event window."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.repository import (
    load_batch_catalog,
    load_images,
    load_sensor_data,
    load_text_records,
)


@dataclass(frozen=True)
class EventEvidence:
    batch_id: str
    start_hr: int
    end_hr: int
    sensors: pd.DataFrame
    sensor_summary: pd.DataFrame
    notes: pd.DataFrame
    images: pd.DataFrame
    outcome: pd.DataFrame


def align_event(batch_id: str, start_hr: int, end_hr: int) -> EventEvidence:
    if start_hr < 0 or end_hr < start_hr:
        raise ValueError("Event window must satisfy 0 <= start_hr <= end_hr")
    sensors = load_sensor_data([batch_id])
    if sensors.empty:
        raise ValueError(f"Unknown batch: {batch_id}")
    sensors = sensors.loc[sensors["time_hr"].between(start_hr, end_hr)].copy()
    notes = load_text_records(batch_id)
    notes = notes.loc[notes["time_hr"].between(start_hr, end_hr)].copy()
    images = load_images(batch_id)
    images = images.loc[images["time_hr"].between(start_hr, end_hr)].copy()
    outcome = load_batch_catalog()
    outcome = outcome.loc[outcome["batch_id"].eq(batch_id)].copy()

    numeric = sensors.select_dtypes("number").drop(columns=["measurement_id", "time_hr"])
    summary = numeric.agg(["mean", "std", "min", "max"]).T.reset_index()
    summary = summary.rename(columns={"index": "variable"})
    return EventEvidence(
        batch_id=batch_id,
        start_hr=start_hr,
        end_hr=end_hr,
        sensors=sensors,
        sensor_summary=summary,
        notes=notes,
        images=images,
        outcome=outcome,
    )
