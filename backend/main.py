"""FastAPI application for trusted tabular and multimodal model inference."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.dependencies import load_registry, require_api_key
from backend.schemas import BatchMetadata, PredictionResponse
from backend.settings import settings
from src.inference.predict import (
    make_multimodal_row,
    make_tabular_row,
    parse_sensor_csv,
)
from src.inference.service import predict_row

app = FastAPI(title="Bioprocess Model Inference API", version="1.0.0")


async def _parse_inputs(metadata: str, sensor_file: UploadFile):
    try:
        parsed_metadata = BatchMetadata.model_validate_json(metadata)
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid metadata: {error}") from error
    content = await sensor_file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Sensor file exceeds upload limit")
    try:
        sensors = parse_sensor_csv(content, parsed_metadata.batch_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return parsed_metadata, sensors


@app.get("/health")
def health() -> dict:
    statuses = {}
    for kind in ("tabular", "multimodal"):
        try:
            _, models = load_registry(kind)
            statuses[kind] = len(models) == 4
        except (OSError, ValueError, json.JSONDecodeError):
            statuses[kind] = False
    return {"status": "ok" if all(statuses.values()) else "degraded", "models": statuses}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def models() -> list[dict]:
    result = []
    for kind in ("tabular", "multimodal"):
        manifest, registered = load_registry(kind)
        result.append(
            {
                "workflow": kind,
                "version": manifest["model_version"],
                "model_ids": sorted(registered),
                "target": "final_titer_g_l",
                "target_unit": "g/L",
                "feature_cutoff_hr": 240,
            }
        )
    return result


@app.post(
    "/v1/predict/tabular",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict_tabular(
    metadata: Annotated[str, Form()], sensor_file: Annotated[UploadFile, File()]
) -> PredictionResponse:
    parsed_metadata, sensors = await _parse_inputs(metadata, sensor_file)
    row = make_tabular_row(parsed_metadata, sensors)
    return predict_row("tabular", row, ["metadata", "sensor"])


@app.post(
    "/v1/predict/multimodal",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict_multimodal(
    metadata: Annotated[str, Form()],
    sensor_file: Annotated[UploadFile, File()],
    notes_file: Annotated[UploadFile | None, File()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
) -> PredictionResponse:
    parsed_metadata, sensors = await _parse_inputs(metadata, sensor_file)
    notes: list[str] = []
    if notes_file:
        content = await notes_file.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Notes file exceeds upload limit")
        try:
            loaded = json.loads(content)
            notes = [item["content"] if isinstance(item, dict) else str(item) for item in loaded]
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            raise HTTPException(status_code=422, detail="Notes must be a JSON list") from error
    uploaded_images = images or []
    if len(uploaded_images) > settings.max_images:
        raise HTTPException(status_code=413, detail="Too many images")
    image_bytes = [await image.read() for image in uploaded_images]
    if any(len(content) > settings.max_upload_bytes for content in image_bytes):
        raise HTTPException(status_code=413, detail="An image exceeds upload limit")
    row = make_multimodal_row(make_tabular_row(parsed_metadata, sensors), notes, image_bytes)
    modalities = ["metadata", "sensor"]
    if notes:
        modalities.append("text")
    if image_bytes:
        modalities.append("image")
    return predict_row("multimodal", row, modalities)
