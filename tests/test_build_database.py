from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.data.build_database import build_database, connect_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


def _build_in_temp(tmp_path: Path) -> Path:
    database_path = tmp_path / "bioprocess.db"
    build_database(
        database_path,
        data_dir=DATA_DIR,
        schema_path=SCHEMA_PATH,
        project_root=PROJECT_ROOT,
    )
    return database_path


def test_database_counts_queries_and_indexes(tmp_path: Path) -> None:
    database_path = _build_in_temp(tmp_path)
    connection = connect_database(database_path, read_only=True)
    try:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("batches", "sensor_data", "outcomes", "text_records", "images")
        }
        assert counts == {
            "batches": 50,
            "sensor_data": 3_050,
            "outcomes": 50,
            "text_records": 200,
            "images": 150,
        }
        b014 = pd.read_sql_query(
            """
            SELECT time_hr, ph, do_pct
            FROM sensor_data
            WHERE batch_id = ?
            ORDER BY time_hr
            """,
            connection,
            params=("B014",),
        )
        assert len(b014) == 61
        assert b014["time_hr"].tolist() == list(range(0, 241, 4))

        joined_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM batches AS b
            JOIN outcomes AS o ON o.batch_id = b.batch_id
            """
        ).fetchone()[0]
        assert joined_count == 50
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        index_names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "idx_sensor_data_batch_time" in index_names
        assert "idx_text_records_batch_time" in index_names
        assert "idx_images_batch_time" in index_names
    finally:
        connection.close()


def test_database_rebuild_is_idempotent(tmp_path: Path) -> None:
    database_path = _build_in_temp(tmp_path)
    first_size = database_path.stat().st_size
    database_path = _build_in_temp(tmp_path)

    connection = connect_database(database_path, read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0] == 3_050
        assert connection.execute("SELECT MAX(measurement_id) FROM sensor_data").fetchone()[0] == 3_050
    finally:
        connection.close()
    assert database_path.stat().st_size == first_size


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    database_path = _build_in_temp(tmp_path)
    connection = connect_database(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO outcomes (
                    batch_id, final_titer_g_l, final_viability_pct,
                    max_vcd_million_ml, quality_metric_score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("B999", 1.0, 90.0, 10.0, 80.0),
            )
    finally:
        connection.close()
