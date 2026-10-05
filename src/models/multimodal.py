"""Batch-level multimodal features and interpretable late-fusion scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.features import TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings"
TABULAR_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "tabular" / "batch_features.csv"


def _pool_embeddings(path: Path, prefix: str) -> pd.DataFrame:
    data = np.load(path)
    batch_ids = data["batch_ids"].astype(str)
    vectors = data["vectors"]
    rows = []
    for batch_id in sorted(set(batch_ids)):
        selected = vectors[batch_ids == batch_id]
        row: dict[str, object] = {
            "batch_id": batch_id,
            f"{prefix}_count": len(selected),
            f"{prefix}_available": 1,
        }
        row.update(
            {f"{prefix}_{index:03d}": value for index, value in enumerate(selected.mean(axis=0))}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_multimodal_feature_table() -> pd.DataFrame:
    tabular = pd.read_csv(TABULAR_FEATURE_PATH)
    text = _pool_embeddings(EMBEDDING_PATH / "text_embeddings.npz", "text")
    images = _pool_embeddings(EMBEDDING_PATH / "image_embeddings.npz", "image")
    fused = tabular.merge(text, on="batch_id", how="left", validate="one_to_one")
    fused = fused.merge(images, on="batch_id", how="left", validate="one_to_one")
    embedding_columns = [
        column
        for column in fused
        if column.startswith(("text_", "image_"))
    ]
    fused[embedding_columns] = fused[embedding_columns].fillna(0)
    return fused


def calculate_risk_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentile-based component scores and a heuristic weighted score."""

    scored = features[["batch_id", TARGET]].copy()
    high_risk = (
        features["do_std_pct"].rank(pct=True)
        + features["lactate_max_g_l"].rank(pct=True)
        + (1 - features["peak_vcd_million_ml"].rank(pct=True))
    ) / 3
    scored["sensor_anomaly"] = high_risk

    for prefix in ("text", "image"):
        columns = [column for column in features if column.startswith(f"{prefix}_") and column[-3:].isdigit()]
        matrix = features[columns].to_numpy()
        centroid = matrix.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        norms = np.linalg.norm(matrix, axis=1)
        similarity = matrix @ centroid / np.where(norms == 0, 1, norms)
        scored[f"{prefix}_anomaly"] = pd.Series(1 - similarity).rank(pct=True).to_numpy()
        scored[f"{prefix}_available"] = features[f"{prefix}_available"].astype(bool)

    weights = {"sensor_anomaly": 0.5, "text_anomaly": 0.2, "image_anomaly": 0.3}
    weighted = np.zeros(len(scored))
    available_weight = np.full(len(scored), weights["sensor_anomaly"])
    weighted += scored["sensor_anomaly"] * weights["sensor_anomaly"]
    for modality in ("text", "image"):
        availability = scored[f"{modality}_available"].astype(float)
        weighted += scored[f"{modality}_anomaly"] * weights[f"{modality}_anomaly"] * availability
        available_weight += weights[f"{modality}_anomaly"] * availability
    scored["multimodal_risk_score"] = weighted / available_weight
    return scored.sort_values("multimodal_risk_score", ascending=False)
