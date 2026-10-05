"""Smoke-test each implemented Streamlit page as a standalone script."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_title"),
    [
        ("views/overview.py", "Bioprocess Multimodal Intelligence Platform"),
        ("pages/1_Batch_Explorer.py", "Batch Explorer"),
        ("pages/2_Batch_Comparison.py", "Batch Comparison"),
        ("pages/3_Data_Quality.py", "Data Quality"),
        ("pages/4_Predictive_Modeling.py", "Predictive Modeling"),
        ("pages/5_Text_Intelligence.py", "Text Intelligence"),
        ("pages/6_Microscopy_Intelligence.py", "Microscopy Intelligence"),
        ("pages/7_Multimodal_Investigation.py", "Multimodal Investigation"),
        ("pages/8_Multimodal_Analytics.py", "Multimodal Analytics"),
        (
            "pages/9_Multimodal_Predictive_Modeling.py",
            "Multimodal Predictive Modeling",
        ),
        ("pages/10_New_Run_Prediction.py", "New Run Prediction"),
        ("pages/11_Scientific_Copilot.py", "Scientific Copilot"),
    ],
)
def test_streamlit_page_runs_without_exception(
    relative_path: str, expected_title: str
) -> None:
    app = AppTest.from_file(PROJECT_ROOT / relative_path)
    app.run(timeout=30)

    assert not app.exception
    assert app.title[0].value == expected_title


def test_navigation_entry_point_runs_overview_by_default() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py")
    app.run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Bioprocess Multimodal Intelligence Platform"
