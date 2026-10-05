"""Trusted artifact registry and optional API-key dependency."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import Header, HTTPException

from backend.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_token and x_api_key != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@lru_cache(maxsize=2)
def load_registry(kind: str) -> tuple[dict, dict[str, object]]:
    if kind not in {"tabular", "multimodal"}:
        raise ValueError(f"Unknown registry: {kind}")
    artifact_dir = PROJECT_ROOT / "artifacts" / kind
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    suffix = "tabular-v1" if kind == "tabular" else "multimodal-v1"
    models = {
        path.name.removesuffix(f"_{suffix}.joblib"): joblib.load(path)
        for path in artifact_dir.glob(f"*_{suffix}.joblib")
    }
    return manifest, models
