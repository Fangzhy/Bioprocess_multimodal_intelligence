"""Tests for built-in new-run prediction examples."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from backend.schemas import BatchMetadata
from src.inference.examples import EXAMPLE_BATCH_ID, load_example_run
from src.inference.predict import REQUIRED_SENSOR_COLUMNS, parse_sensor_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_run_matches_api_input_contract() -> None:
    example = load_example_run()
    metadata = BatchMetadata.model_validate(example.metadata)
    sensor_csv = example.sensor_data.to_csv(index=False).encode("utf-8")
    parsed_sensor = parse_sensor_csv(sensor_csv, metadata.batch_id)

    assert metadata.batch_id == EXAMPLE_BATCH_ID
    assert len(parsed_sensor) == 61
    assert REQUIRED_SENSOR_COLUMNS.issubset(parsed_sensor.columns)
    assert "titer_g_l" not in example.sensor_data
    assert len(example.notes) == 4
    assert len(example.image_paths) == 3
    assert all(path.is_file() for path in example.image_paths)
    assert isinstance(json.loads(example.notes.to_json(orient="records")), list)


def test_example_buttons_load_tabular_and_multimodal_forms() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "pages" / "10_New_Run_Prediction.py")
    app.run(timeout=30)

    app.button[0].click().run(timeout=30)
    assert app.radio[0].value == "Built-in example"
    assert app.radio[1].value == "Tabular baseline"
    assert "NEW_DEMO_001" in app.text_area[0].value

    app.button[1].click().run(timeout=30)
    assert app.radio[0].value == "Built-in example"
    assert app.radio[1].value == "Multimodal"
    assert any(
        "Synthetic microscopy images" in markdown.value for markdown in app.markdown
    )
