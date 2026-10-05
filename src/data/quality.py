"""Data-quality summaries for the synthetic bioprocess dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SENSOR_RANGES = {
    "temperature_c": (30.0, 42.0),
    "ph": (6.5, 7.5),
    "do_pct": (0.0, 100.0),
    "agitation_rpm": (0.0, 1_500.0),
    "air_flow_slpm": (0.0, 10.0),
    "o2_flow_slpm": (0.0, 10.0),
    "co2_flow_slpm": (0.0, 10.0),
    "feed_rate_ml_hr": (0.0, 100.0),
    "glucose_g_l": (0.0, 20.0),
    "lactate_g_l": (0.0, 20.0),
    "viable_cell_density_million_ml": (0.0, 100.0),
    "viability_pct": (0.0, 100.0),
    "titer_g_l": (0.0, 20.0),
}


@dataclass(frozen=True)
class QualityReport:
    missing_values: pd.DataFrame
    duplicates: pd.DataFrame
    range_violations: pd.DataFrame
    outliers: pd.DataFrame
    batch_coverage: pd.DataFrame

    @property
    def total_missing(self) -> int:
        return int(self.missing_values["missing_count"].sum())

    @property
    def total_duplicates(self) -> int:
        return int(self.duplicates["duplicate_count"].sum())

    @property
    def total_range_violations(self) -> int:
        return int(self.range_violations["violation_count"].sum())

    @property
    def total_outliers(self) -> int:
        return int(self.outliers["outlier_count"].sum())


def _missing_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            count = int(frame[column].isna().sum())
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "missing_count": count,
                    "missing_pct": 100 * count / len(frame) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _duplicate_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = {
        "batches": ["batch_id"],
        "sensor_data": ["batch_id", "time_hr"],
        "outcomes": ["batch_id"],
        "text_records": ["text_id"],
        "images": ["image_id"],
    }
    return pd.DataFrame(
        [
            {
                "table": table_name,
                "key": ", ".join(key_columns),
                "duplicate_count": int(frame.duplicated(key_columns).sum()),
            }
            for table_name, key_columns in keys.items()
            if (frame := tables[table_name]) is not None
        ]
    )


def _range_summary(sensor_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, (minimum, maximum) in SENSOR_RANGES.items():
        violations = ~sensor_data[column].between(minimum, maximum)
        rows.append(
            {
                "column": column,
                "expected_min": minimum,
                "expected_max": maximum,
                "observed_min": float(sensor_data[column].min()),
                "observed_max": float(sensor_data[column].max()),
                "violation_count": int(violations.sum()),
            }
        )
    return pd.DataFrame(rows)


def _outlier_summary(sensor_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SENSOR_RANGES:
        first_quartile = float(sensor_data[column].quantile(0.25))
        third_quartile = float(sensor_data[column].quantile(0.75))
        interquartile_range = third_quartile - first_quartile
        lower = first_quartile - 1.5 * interquartile_range
        upper = third_quartile + 1.5 * interquartile_range
        outliers = ~sensor_data[column].between(lower, upper)
        rows.append(
            {
                "column": column,
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "outlier_count": int(outliers.sum()),
                "outlier_pct": 100 * float(outliers.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_quality_report(tables: dict[str, pd.DataFrame]) -> QualityReport:
    missing_tables = {
        "batches",
        "sensor_data",
        "outcomes",
        "text_records",
        "images",
    } - set(tables)
    if missing_tables:
        raise ValueError(f"Required tables are missing: {sorted(missing_tables)}")

    sensor_data = tables["sensor_data"]
    coverage = (
        sensor_data.groupby("batch_id", as_index=False)
        .agg(
            measurement_count=("time_hr", "size"),
            first_time_hr=("time_hr", "min"),
            last_time_hr=("time_hr", "max"),
        )
        .sort_values("batch_id")
    )
    return QualityReport(
        missing_values=_missing_summary(tables),
        duplicates=_duplicate_summary(tables),
        range_violations=_range_summary(sensor_data),
        outliers=_outlier_summary(sensor_data),
        batch_coverage=coverage,
    )
