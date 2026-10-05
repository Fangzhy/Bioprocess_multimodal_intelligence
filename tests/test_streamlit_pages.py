"""Smoke-test each implemented Streamlit page as a standalone script."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_title"),
    [
        ("app.py", "Bioprocess Multimodal Intelligence Platform"),
        ("pages/1_Batch_Explorer.py", "Batch Explorer"),
        ("pages/2_Batch_Comparison.py", "Batch Comparison"),
        ("pages/3_Data_Quality.py", "Data Quality"),
        ("pages/4_Predictive_Modeling.py", "Predictive Modeling"),
    ],
)
def test_streamlit_page_runs_without_exception(
    relative_path: str, expected_title: str
) -> None:
    app = AppTest.from_file(PROJECT_ROOT / relative_path)
    app.run(timeout=30)

    assert not app.exception
    assert app.title[0].value == expected_title
