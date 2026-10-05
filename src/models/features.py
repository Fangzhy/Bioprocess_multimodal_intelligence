"""Batch-level feature engineering for tabular titer models."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["cell_line", "media_type", "feed_strategy"]
NUMERIC_FEATURES = [
    "bioreactor_scale_l",
    "seed_density_million_ml",
    "temperature_mean_c",
    "temperature_std_c",
    "ph_mean",
    "ph_std",
    "do_mean_pct",
    "do_std_pct",
    "do_min_pct",
    "agitation_mean_rpm",
    "agitation_max_rpm",
    "air_flow_mean_slpm",
    "o2_flow_mean_slpm",
    "co2_flow_mean_slpm",
    "total_feed_l",
    "glucose_min_g_l",
    "glucose_final_g_l",
    "lactate_max_g_l",
    "lactate_final_g_l",
    "peak_vcd_million_ml",
    "time_to_peak_vcd_hr",
    "final_vcd_million_ml",
    "vcd_auc_million_cell_hr_ml",
    "growth_rate_per_hr",
    "viability_min_pct",
    "viability_final_pct",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "final_titer_g_l"


def _trapezoid(y: pd.Series, x: pd.Series) -> float:
    return float(np.trapezoid(y.to_numpy(), x.to_numpy()))


def _growth_rate(sensor_data: pd.DataFrame) -> float:
    """Estimate exponential-phase growth from log(VCD), using 24-96 hours."""

    growth_window = sensor_data.loc[sensor_data["time_hr"].between(24, 96)]
    positive = growth_window.loc[
        growth_window["viable_cell_density_million_ml"].gt(0)
    ]
    if len(positive) < 2:
        return float("nan")
    slope, _ = np.polyfit(
        positive["time_hr"],
        np.log(positive["viable_cell_density_million_ml"]),
        deg=1,
    )
    return float(slope)


def engineer_batch_features(
    batches: pd.DataFrame,
    sensor_data: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one modeling row per batch without direct titer-derived predictors."""

    rows: list[dict[str, object]] = []
    metadata = batches.set_index("batch_id")
    for batch_id, measurements in sensor_data.groupby("batch_id", sort=True):
        measurements = measurements.sort_values("time_hr")
        batch = metadata.loc[batch_id]
        peak_index = measurements["viable_cell_density_million_ml"].idxmax()
        rows.append(
            {
                "batch_id": batch_id,
                "bioreactor_scale_l": batch["bioreactor_scale_l"],
                "seed_density_million_ml": batch["seed_density_million_ml"],
                "cell_line": batch["cell_line"],
                "media_type": batch["media_type"],
                "feed_strategy": batch["feed_strategy"],
                "temperature_mean_c": measurements["temperature_c"].mean(),
                "temperature_std_c": measurements["temperature_c"].std(),
                "ph_mean": measurements["ph"].mean(),
                "ph_std": measurements["ph"].std(),
                "do_mean_pct": measurements["do_pct"].mean(),
                "do_std_pct": measurements["do_pct"].std(),
                "do_min_pct": measurements["do_pct"].min(),
                "agitation_mean_rpm": measurements["agitation_rpm"].mean(),
                "agitation_max_rpm": measurements["agitation_rpm"].max(),
                "air_flow_mean_slpm": measurements["air_flow_slpm"].mean(),
                "o2_flow_mean_slpm": measurements["o2_flow_slpm"].mean(),
                "co2_flow_mean_slpm": measurements["co2_flow_slpm"].mean(),
                "total_feed_l": _trapezoid(
                    measurements["feed_rate_ml_hr"], measurements["time_hr"]
                )
                / 1_000,
                "glucose_min_g_l": measurements["glucose_g_l"].min(),
                "glucose_final_g_l": measurements["glucose_g_l"].iloc[-1],
                "lactate_max_g_l": measurements["lactate_g_l"].max(),
                "lactate_final_g_l": measurements["lactate_g_l"].iloc[-1],
                "peak_vcd_million_ml": measurements[
                    "viable_cell_density_million_ml"
                ].max(),
                "time_to_peak_vcd_hr": measurements.loc[peak_index, "time_hr"],
                "final_vcd_million_ml": measurements[
                    "viable_cell_density_million_ml"
                ].iloc[-1],
                "vcd_auc_million_cell_hr_ml": _trapezoid(
                    measurements["viable_cell_density_million_ml"],
                    measurements["time_hr"],
                ),
                "growth_rate_per_hr": _growth_rate(measurements),
                "viability_min_pct": measurements["viability_pct"].min(),
                "viability_final_pct": measurements["viability_pct"].iloc[-1],
            }
        )

    features = pd.DataFrame(rows)
    if outcomes is not None:
        features = features.merge(
            outcomes[["batch_id", TARGET]],
            on="batch_id",
            how="inner",
            validate="one_to_one",
        )
    return features
