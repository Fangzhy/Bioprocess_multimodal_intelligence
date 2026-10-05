"""Build the SQLite demonstration database from Milestone 2 CSV files."""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
TABLE_FILES = {
    "batches": "batches.csv",
    "sensor_data": "sensor_data.csv",
    "outcomes": "outcomes.csv",
    "text_records": "text_records.csv",
    "images": "images.csv",
}

EXPECTED_COLUMNS = {
    "batches": (
        "batch_id",
        "cell_line",
        "media_type",
        "media_lot",
        "bioreactor_scale_l",
        "seed_density_million_ml",
        "feed_strategy",
        "experiment_date",
    ),
    "sensor_data": (
        "batch_id",
        "time_hr",
        "temperature_c",
        "ph",
        "do_pct",
        "agitation_rpm",
        "air_flow_slpm",
        "o2_flow_slpm",
        "co2_flow_slpm",
        "feed_rate_ml_hr",
        "glucose_g_l",
        "lactate_g_l",
        "viable_cell_density_million_ml",
        "viability_pct",
        "titer_g_l",
    ),
    "outcomes": (
        "batch_id",
        "final_titer_g_l",
        "final_viability_pct",
        "max_vcd_million_ml",
        "quality_metric_score",
    ),
    "text_records": ("text_id", "batch_id", "time_hr", "text_type", "content"),
    "images": (
        "image_id",
        "batch_id",
        "time_hr",
        "file_path",
        "image_type",
        "is_synthetic",
    ),
}


def connect_database(database_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open SQLite with foreign keys enabled and named-column rows."""

    database_path = Path(database_path).resolve()
    if read_only:
        connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _load_source_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for table_name, file_name in TABLE_FILES.items():
        file_path = data_dir / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Required source file is missing: {file_path}")
        frame = pd.read_csv(file_path)
        expected = list(EXPECTED_COLUMNS[table_name])
        if list(frame.columns) != expected:
            raise ValueError(
                f"{file_name} columns do not match the schema. "
                f"Expected {expected}; received {list(frame.columns)}"
            )
        if frame.isna().any().any():
            raise ValueError(f"{file_name} contains missing values")
        tables[table_name] = frame
    return tables


def _validate_source_relationships(
    tables: dict[str, pd.DataFrame], project_root: Path
) -> None:
    batches = tables["batches"]
    batch_ids = set(batches["batch_id"])
    if not batches["batch_id"].is_unique:
        raise ValueError("batches.csv contains duplicate batch_id values")

    for table_name in ("sensor_data", "outcomes", "text_records", "images"):
        unknown_ids = set(tables[table_name]["batch_id"]) - batch_ids
        if unknown_ids:
            raise ValueError(f"{table_name} references unknown batches: {unknown_ids}")

    if tables["sensor_data"].duplicated(["batch_id", "time_hr"]).any():
        raise ValueError("sensor_data.csv contains duplicate batch/time measurements")
    if not tables["outcomes"]["batch_id"].is_unique:
        raise ValueError("outcomes.csv contains duplicate batch outcomes")
    if not tables["text_records"]["text_id"].is_unique:
        raise ValueError("text_records.csv contains duplicate text_id values")
    if not tables["images"]["image_id"].is_unique:
        raise ValueError("images.csv contains duplicate image_id values")

    missing_images = [
        file_path
        for file_path in tables["images"]["file_path"]
        if not (project_root / file_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Image files are missing: {missing_images[:5]}")


def _sql_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rows(frame: pd.DataFrame) -> Iterable[tuple[Any, ...]]:
    for row in frame.itertuples(index=False, name=None):
        yield tuple(_sql_value(value) for value in row)


def _sqlite_boolean(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return 1
        if normalized in {"false", "0", "no"}:
            return 0
        raise ValueError(f"Cannot interpret boolean value: {value!r}")
    return int(bool(value))


def _insert_tables(
    connection: sqlite3.Connection, tables: dict[str, pd.DataFrame]
) -> None:
    connection.executemany(
        """
        INSERT INTO batches (
            batch_id, cell_line, media_type, media_lot, bioreactor_scale_l,
            seed_density_million_ml, feed_strategy, experiment_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _rows(tables["batches"]),
    )

    sensor_rows = (
        (measurement_id, *row)
        for measurement_id, row in enumerate(_rows(tables["sensor_data"]), start=1)
    )
    connection.executemany(
        """
        INSERT INTO sensor_data (
            measurement_id, batch_id, time_hr, temperature_c, ph, do_pct,
            agitation_rpm, air_flow_slpm, o2_flow_slpm, co2_flow_slpm,
            feed_rate_ml_hr, glucose_g_l, lactate_g_l,
            viable_cell_density_million_ml, viability_pct, titer_g_l
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sensor_rows,
    )
    connection.executemany(
        """
        INSERT INTO outcomes (
            batch_id, final_titer_g_l, final_viability_pct,
            max_vcd_million_ml, quality_metric_score
        ) VALUES (?, ?, ?, ?, ?)
        """,
        _rows(tables["outcomes"]),
    )
    connection.executemany(
        """
        INSERT INTO text_records (
            text_id, batch_id, time_hr, text_type, content
        ) VALUES (?, ?, ?, ?, ?)
        """,
        _rows(tables["text_records"]),
    )
    image_rows = (
        (*row[:-1], _sqlite_boolean(row[-1])) for row in _rows(tables["images"])
    )
    connection.executemany(
        """
        INSERT INTO images (
            image_id, batch_id, time_hr, file_path, image_type, is_synthetic
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        image_rows,
    )


def build_database(
    database_path: Path,
    *,
    data_dir: Path,
    schema_path: Path,
    project_root: Path,
) -> dict[str, int]:
    """Build and atomically replace a SQLite snapshot from source CSV files."""

    database_path = Path(database_path).resolve()
    data_dir = Path(data_dir).resolve()
    schema_path = Path(schema_path).resolve()
    project_root = Path(project_root).resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file is missing: {schema_path}")

    tables = _load_source_tables(data_dir)
    _validate_source_relationships(tables, project_root)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    connection: sqlite3.Connection | None = None
    try:
        connection = connect_database(temporary_path)
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        with connection:
            _insert_tables(connection, tables)

        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise sqlite3.IntegrityError(
                f"Foreign-key validation failed: {foreign_key_errors}"
            )
        integrity_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity_result != "ok":
            raise sqlite3.DatabaseError(f"SQLite integrity check failed: {integrity_result}")
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version != SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"Expected schema version {SCHEMA_VERSION}, found {schema_version}"
            )
        connection.close()
        connection = None
        os.replace(temporary_path, database_path)
    except Exception:
        if connection is not None:
            connection.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return {table_name: len(frame) for table_name, frame in tables.items()}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=project_root / "database" / "bioprocess.db",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data" / "raw"
    )
    parser.add_argument(
        "--schema", type=Path, default=project_root / "database" / "schema.sql"
    )
    args = parser.parse_args()
    counts = build_database(
        args.database,
        data_dir=args.data_dir,
        schema_path=args.schema,
        project_root=project_root,
    )
    count_summary = ", ".join(f"{name}={count}" for name, count in counts.items())
    print(f"Built {args.database.resolve()} ({count_summary})")


if __name__ == "__main__":
    main()
