"""Shared prediction service used by FastAPI and the hosted Streamlit demo."""

from __future__ import annotations

import json

import numpy as np
from pydantic import ValidationError

from backend.dependencies import load_registry
from backend.schemas import BatchMetadata, ModelPrediction, PredictionResponse
from src.inference.predict import (
    make_multimodal_row,
    make_tabular_row,
    parse_sensor_csv,
)


def predict_row(kind: str, row, modalities: list[str]) -> PredictionResponse:
    """Run every trusted model registered for one workflow."""

    manifest, models = load_registry(kind)
    threshold = manifest.get("low_titer_threshold_g_l")
    predictions = []
    for model_id, pipeline in sorted(models.items()):
        value = float(np.asarray(pipeline.predict(row)).reshape(-1)[0])
        predictions.append(
            ModelPrediction(
                model_id=model_id,
                predicted_final_titer_g_l=value,
                low_titer=value < threshold if threshold is not None else None,
            )
        )
    missing = [item for item in ["text", "image"] if item not in modalities]
    return PredictionResponse(
        model_version=manifest.get("model_version", "unknown"),
        workflow=kind,
        modalities_present=modalities,
        missing_modalities=missing if kind == "multimodal" else [],
        low_titer_threshold_g_l=threshold,
        predictions=predictions,
        warnings=["End-of-run model using measurements through 240 hours."],
    )


def predict_request(
    workflow: str,
    metadata_content: bytes,
    sensor_content: bytes,
    notes_content: bytes | None = None,
    image_contents: list[bytes] | None = None,
    text_embedding: np.ndarray | None = None,
    image_embedding: np.ndarray | None = None,
) -> PredictionResponse:
    """Validate raw request files and predict without an HTTP round trip."""

    if workflow not in {"tabular", "multimodal"}:
        raise ValueError(f"Unsupported workflow: {workflow}")
    try:
        metadata = BatchMetadata.model_validate_json(metadata_content)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"Invalid metadata: {error}") from error
    sensors = parse_sensor_csv(sensor_content, metadata.batch_id)
    tabular_row = make_tabular_row(metadata, sensors)
    if workflow == "tabular":
        return predict_row("tabular", tabular_row, ["metadata", "sensor"])

    notes: list[str] = []
    if notes_content:
        try:
            loaded = json.loads(notes_content)
            if not isinstance(loaded, list):
                raise TypeError
            notes = [
                item["content"] if isinstance(item, dict) else str(item)
                for item in loaded
            ]
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            raise ValueError("Notes must be a JSON list") from error
    images = image_contents or []
    row = make_multimodal_row(
        tabular_row,
        notes,
        images,
        text_vector=text_embedding,
        image_vector=image_embedding,
    )
    modalities = ["metadata", "sensor"]
    if notes:
        modalities.append("text")
    if images:
        modalities.append("image")
    return predict_row("multimodal", row, modalities)
