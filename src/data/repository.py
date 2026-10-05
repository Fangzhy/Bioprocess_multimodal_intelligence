"""Read-only query functions used by the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_database import connect_database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "bioprocess.db"
TABLE_NAMES = ("batches", "sensor_data", "outcomes", "text_records", "images")


def _read_query(
    sql: str,
    params: Sequence[Any] = (),
    *,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    connection = connect_database(database_path, read_only=True)
    try:
        return pd.read_sql_query(sql, connection, params=tuple(params))
    finally:
        connection.close()


def load_batches(*, database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    return _read_query(
        "SELECT * FROM batches ORDER BY batch_id", database_path=database_path
    )


def load_outcomes(*, database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    return _read_query(
        "SELECT * FROM outcomes ORDER BY batch_id", database_path=database_path
    )


def load_sensor_data(
    batch_ids: Sequence[str] | None = None,
    *,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    if not batch_ids:
        return _read_query(
            "SELECT * FROM sensor_data ORDER BY batch_id, time_hr",
            database_path=database_path,
        )
    placeholders = ", ".join("?" for _ in batch_ids)
    return _read_query(
        f"""
        SELECT * FROM sensor_data
        WHERE batch_id IN ({placeholders})
        ORDER BY batch_id, time_hr
        """,
        batch_ids,
        database_path=database_path,
    )


def load_text_records(
    batch_id: str | None = None,
    *,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    if batch_id is None:
        return _read_query(
            "SELECT * FROM text_records ORDER BY batch_id, time_hr, text_id",
            database_path=database_path,
        )
    return _read_query(
        """
        SELECT * FROM text_records
        WHERE batch_id = ?
        ORDER BY time_hr, text_id
        """,
        (batch_id,),
        database_path=database_path,
    )


def load_images(
    batch_id: str | None = None,
    *,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
    if batch_id is None:
        return _read_query(
            "SELECT * FROM images ORDER BY batch_id, time_hr, image_id",
            database_path=database_path,
        )
    return _read_query(
        """
        SELECT * FROM images
        WHERE batch_id = ?
        ORDER BY time_hr, image_id
        """,
        (batch_id,),
        database_path=database_path,
    )


def load_batch_catalog(*, database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    return _read_query(
        """
        SELECT
            b.*,
            o.final_titer_g_l,
            o.final_viability_pct,
            o.max_vcd_million_ml,
            o.quality_metric_score
        FROM batches AS b
        JOIN outcomes AS o ON o.batch_id = b.batch_id
        ORDER BY b.batch_id
        """,
        database_path=database_path,
    )


def load_table(
    table_name: str, *, database_path: Path = DEFAULT_DATABASE_PATH
) -> pd.DataFrame:
    """Load an allowed table; table names cannot come from arbitrary SQL input."""

    if table_name not in TABLE_NAMES:
        raise ValueError(f"Unsupported table: {table_name}")
    return _read_query(
        f"SELECT * FROM {table_name}", database_path=database_path
    )


def select_reference_batches(
    catalog: pd.DataFrame,
    selected_batch_id: str,
    cohort: str,
) -> list[str]:
    """Select a transparent metadata-based historical reference cohort."""

    if selected_batch_id not in set(catalog["batch_id"]):
        raise ValueError(f"Unknown batch: {selected_batch_id}")
    selected = catalog.loc[catalog["batch_id"].eq(selected_batch_id)].iloc[0]
    candidates = catalog.loc[~catalog["batch_id"].eq(selected_batch_id)]

    if cohort == "Same media and cell line":
        references = candidates.loc[
            candidates["media_type"].eq(selected["media_type"])
            & candidates["cell_line"].eq(selected["cell_line"])
        ]
    elif cohort == "Same media":
        references = candidates.loc[
            candidates["media_type"].eq(selected["media_type"])
        ]
    elif cohort == "All other batches":
        references = candidates
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    return references["batch_id"].tolist()
