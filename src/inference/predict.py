"""Transform uploaded raw runs and execute trusted fitted pipelines."""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from backend.schemas import BatchMetadata
from src.embeddings.store import encode_texts, load_image_encoder, load_text_encoder
from src.models.features import MODEL_FEATURES, engineer_batch_features

REQUIRED_SENSOR_COLUMNS = {
    "time_hr",
    "temperature_c",
    "ph",
    "do_pct",
    "agitation_rpm",
    "air_flow_slpm",
    "o2_flow_slpm",
    "co2_flow_slpm",
    "feed_rate_ml_hr",
    "glucose_g_l",
    "lactate_g_l",
    "viable_cell_density_million_ml",
    "viability_pct",
}


def parse_sensor_csv(content: bytes, batch_id: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        raise ValueError("Sensor file must be a readable CSV") from error
    missing = REQUIRED_SENSOR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Sensor CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Sensor CSV must contain measurements")
    if frame["time_hr"].min() < 0 or frame["time_hr"].max() > 240:
        raise ValueError("Sensor timestamps must be between 0 and 240 hours")
    if frame["time_hr"].duplicated().any():
        raise ValueError("Sensor timestamps must be unique")
    frame["batch_id"] = batch_id
    return frame.sort_values("time_hr")


def make_tabular_row(metadata: BatchMetadata, sensor_data: pd.DataFrame) -> pd.DataFrame:
    batch = pd.DataFrame([metadata.model_dump()])
    return engineer_batch_features(batch, sensor_data)[MODEL_FEATURES]


@lru_cache(maxsize=1)
def _text_encoder():
    try:
        return load_text_encoder(local_files_only=True)
    except OSError:
        return load_text_encoder(local_files_only=False)


@lru_cache(maxsize=1)
def _image_encoder():
    return load_image_encoder()


def make_multimodal_row(
    tabular_row: pd.DataFrame,
    notes: list[str],
    image_bytes: list[bytes],
    *,
    text_vector: np.ndarray | None = None,
    image_vector: np.ndarray | None = None,
) -> pd.DataFrame:
    if text_vector is not None:
        text_vector = np.asarray(text_vector, dtype=np.float32)
    elif notes:
        text_vector = encode_texts(notes, _text_encoder()).mean(axis=0)
    else:
        text_vector = np.zeros(384, dtype=np.float32)

    if image_vector is not None:
        image_vector = np.asarray(image_vector, dtype=np.float32)
    elif image_bytes:
        model, preprocess = _image_encoder()
        tensors = []
        import torch

        with torch.inference_mode():
            for content in image_bytes:
                with Image.open(io.BytesIO(content)) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors)
            vectors = model.encode_image(batch)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
            image_vector = vectors.cpu().numpy().mean(axis=0)
    else:
        image_vector = np.zeros(512, dtype=np.float32)
    modality_values = {
        "text_count": len(notes),
        "text_available": int(bool(notes)),
        "image_count": len(image_bytes),
        "image_available": int(bool(image_bytes)),
    }
    modality_values.update(
        {f"text_{index:03d}": value for index, value in enumerate(text_vector)}
    )
    modality_values.update(
        {f"image_{index:03d}": value for index, value in enumerate(image_vector)}
    )
    modality_frame = pd.DataFrame([modality_values], index=tabular_row.index)
    return pd.concat([tabular_row, modality_frame], axis=1)
