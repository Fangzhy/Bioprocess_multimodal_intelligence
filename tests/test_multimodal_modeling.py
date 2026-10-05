"""Tests for heuristic and trained multimodal fusion."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.features import TARGET
from src.models.multimodal import build_multimodal_feature_table, calculate_risk_scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "multimodal"


def test_fused_feature_table_has_expected_embedding_dimensions() -> None:
    frame = build_multimodal_feature_table()
    text_columns = [column for column in frame if column.startswith("text_") and column[-3:].isdigit()]
    image_columns = [column for column in frame if column.startswith("image_") and column[-3:].isdigit()]
    assert len(frame) == 50
    assert len(text_columns) == 384
    assert len(image_columns) == 512
    assert frame["text_count"].eq(4).all()
    assert frame["image_count"].eq(3).all()


def test_heuristic_scores_are_bounded_and_ranked() -> None:
    scores = calculate_risk_scores(build_multimodal_feature_table())
    components = ["sensor_anomaly", "text_anomaly", "image_anomaly", "multimodal_risk_score"]
    assert scores[components].apply(lambda column: column.between(0, 1).all()).all()
    assert scores["multimodal_risk_score"].is_monotonic_decreasing


def test_multimodal_experiment_reuses_holdout_and_fits_pca() -> None:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    tabular_manifest = json.loads(
        (PROJECT_ROOT / "artifacts" / "tabular" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["test_batch_ids"] == tabular_manifest["split"]["test_batch_ids"]
    assert manifest["pca"]["requested_components"] == 20

    features = pd.read_csv(ARTIFACT_DIR / "fused_batch_features.csv")
    exported = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    test = features.loc[features["batch_id"].isin(manifest["test_batch_ids"])]
    pipeline = joblib.load(ARTIFACT_DIR / "random_forest_multimodal-v1.joblib")
    predicted = pipeline.predict(test.drop(columns=[TARGET, "batch_id"]))
    expected = exported.loc[
        exported["model"].eq("Random Forest")
        & exported["feature_set"].eq("Tabular + text + image"),
        "predicted_titer_g_l",
    ]
    assert np.allclose(predicted, expected)
    assert pipeline.named_steps["preprocessor"].named_transformers_["text"].named_steps[
        "pca"
    ].n_components_ == 20
