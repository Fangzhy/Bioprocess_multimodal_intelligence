"""Built-in synthetic request examples for the prediction interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.repository import (
    PROJECT_ROOT,
    load_batch_catalog,
    load_images,
    load_sensor_data,
    load_text_records,
)

EXAMPLE_SOURCE_BATCH = "B018"
EXAMPLE_BATCH_ID = "NEW_DEMO_001"


@dataclass(frozen=True)
class ExampleRun:
    metadata: dict[str, str | float]
    sensor_data: pd.DataFrame
    notes: pd.DataFrame
    image_paths: tuple[Path, ...]
    text_embedding: np.ndarray
    image_embedding: np.ndarray


def _pooled_embedding(file_name: str, batch_id: str) -> np.ndarray:
    stored = np.load(PROJECT_ROOT / "data" / "processed" / "embeddings" / file_name)
    selected = stored["vectors"][stored["batch_ids"].astype(str) == batch_id]
    if not len(selected):
        raise ValueError(f"No stored {file_name} vectors for {batch_id}")
    return selected.mean(axis=0).astype(np.float32)


def load_example_run() -> ExampleRun:
    """Create a new-run payload from one reproducible synthetic demonstration run."""

    catalog = load_batch_catalog()
    source = catalog.loc[catalog["batch_id"].eq(EXAMPLE_SOURCE_BATCH)].iloc[0]
    metadata = {
        "batch_id": EXAMPLE_BATCH_ID,
        "cell_line": str(source["cell_line"]),
        "media_type": str(source["media_type"]),
        "bioreactor_scale_l": float(source["bioreactor_scale_l"]),
        "seed_density_million_ml": float(source["seed_density_million_ml"]),
        "feed_strategy": str(source["feed_strategy"]),
    }
    sensor_data = load_sensor_data([EXAMPLE_SOURCE_BATCH]).drop(
        columns=["measurement_id", "batch_id", "titer_g_l"]
    )
    notes = load_text_records(EXAMPLE_SOURCE_BATCH)[
        ["time_hr", "text_type", "content"]
    ].copy()
    image_paths = tuple(
        PROJECT_ROOT / relative_path
        for relative_path in load_images(EXAMPLE_SOURCE_BATCH)["file_path"]
    )
    return ExampleRun(
        metadata=metadata,
        sensor_data=sensor_data,
        notes=notes,
        image_paths=image_paths,
        text_embedding=_pooled_embedding("text_embeddings.npz", EXAMPLE_SOURCE_BATCH),
        image_embedding=_pooled_embedding("image_embeddings.npz", EXAMPLE_SOURCE_BATCH),
    )
