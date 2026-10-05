"""Integration tests for versioned FastAPI model inference."""

import json

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.data.repository import load_batch_catalog, load_sensor_data

client = TestClient(app)


def _request_files(batch_id: str = "B018") -> tuple[dict, dict]:
    catalog = load_batch_catalog()
    row = catalog.loc[catalog["batch_id"].eq(batch_id)].iloc[0]
    metadata = {
        "batch_id": batch_id,
        "cell_line": row["cell_line"],
        "media_type": row["media_type"],
        "bioreactor_scale_l": row["bioreactor_scale_l"],
        "seed_density_million_ml": row["seed_density_million_ml"],
        "feed_strategy": row["feed_strategy"],
    }
    sensors = load_sensor_data([batch_id]).drop(columns=["measurement_id", "batch_id"])
    return {"metadata": json.dumps(metadata)}, {
        "sensor_file": ("sensor.csv", sensors.to_csv(index=False), "text/csv")
    }


def test_health_and_model_registry() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert {record["workflow"] for record in models.json()} == {"tabular", "multimodal"}


def test_tabular_and_multimodal_predictions() -> None:
    data, files = _request_files()
    tabular = client.post("/v1/predict/tabular", data=data, files=files)
    assert tabular.status_code == 200
    assert len(tabular.json()["predictions"]) == 4
    assert tabular.json()["model_version"] == "tabular-v1"
    exported = pd.read_csv("artifacts/tabular/test_predictions.csv")
    expected = exported.loc[
        exported["batch_id"].eq("B018")
        & exported["model"].eq("Random Forest"),
        "predicted_titer_g_l",
    ].iloc[0]
    returned = next(
        item["predicted_final_titer_g_l"]
        for item in tabular.json()["predictions"]
        if item["model_id"] == "random_forest"
    )
    assert returned == expected

    data, files = _request_files()
    multimodal = client.post("/v1/predict/multimodal", data=data, files=files)
    assert multimodal.status_code == 200
    payload = multimodal.json()
    assert len(payload["predictions"]) == 4
    assert payload["missing_modalities"] == ["text", "image"]


def test_api_rejects_missing_sensor_columns() -> None:
    data, _ = _request_files()
    files = {"sensor_file": ("sensor.csv", pd.DataFrame({"time_hr": [0]}).to_csv(index=False), "text/csv")}
    response = client.post("/v1/predict/tabular", data=data, files=files)
    assert response.status_code == 422
    assert "missing columns" in response.json()["detail"]
