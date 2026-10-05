"""Versioned API request and response schemas."""

from pydantic import BaseModel, Field


class BatchMetadata(BaseModel):
    batch_id: str = Field(min_length=1, max_length=80)
    cell_line: str
    media_type: str
    bioreactor_scale_l: float = Field(gt=0)
    seed_density_million_ml: float = Field(gt=0)
    feed_strategy: str


class ModelPrediction(BaseModel):
    model_id: str
    predicted_final_titer_g_l: float
    low_titer: bool | None = None


class PredictionResponse(BaseModel):
    api_version: str = "v1"
    model_version: str
    workflow: str
    target_unit: str = "g/L"
    feature_cutoff_hr: int = 240
    modalities_present: list[str]
    missing_modalities: list[str]
    low_titer_threshold_g_l: float | None = None
    predictions: list[ModelPrediction]
    warnings: list[str] = Field(default_factory=list)
