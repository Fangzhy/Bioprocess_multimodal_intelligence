"""Tests for Milestone 5 batch features and saved model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.repository import load_batches, load_outcomes, load_sensor_data
from src.models.features import MODEL_FEATURES, TARGET, engineer_batch_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tabular"


def test_batch_features_are_one_row_per_batch_without_target_leakage() -> None:
    outcomes = load_outcomes()
    features = engineer_batch_features(load_batches(), load_sensor_data(), outcomes)

    assert len(features) == 50
    assert features["batch_id"].is_unique
    assert features[MODEL_FEATURES].notna().all().all()
    assert not any("titer" in feature for feature in MODEL_FEATURES)
    assert features["total_feed_l"].gt(0).all()
    assert np.allclose(
        features.sort_values("batch_id")[TARGET],
        outcomes.sort_values("batch_id")[TARGET],
    )


def test_saved_artifacts_match_exported_holdout_predictions() -> None:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    features = pd.read_csv(ARTIFACT_DIR / "batch_features.csv")
    exported = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")

    train_ids = set(manifest["split"]["train_batch_ids"])
    test_ids = set(manifest["split"]["test_batch_ids"])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 40
    assert len(test_ids) == 10

    test = features.loc[features["batch_id"].isin(test_ids)]
    for model_record in manifest["models"]:
        pipeline = joblib.load(ARTIFACT_DIR / model_record["artifact"])
        predictions = np.asarray(pipeline.predict(test[MODEL_FEATURES])).reshape(-1)
        expected = exported.loc[
            exported["model"].eq(model_record["model"]), "predicted_titer_g_l"
        ].to_numpy()
        assert np.allclose(predictions, expected)
