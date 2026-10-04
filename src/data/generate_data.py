"""Generate reproducible synthetic data for the bioprocess learning project.

The equations in this module are illustrative. They create coherent signals for
learning data engineering and machine-learning workflows; they are not a
validated mechanistic model of a real cell-culture process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

GENERATOR_VERSION = "1.0.0"

SCENARIOS = {
    "B007": ("high_lactate", 72, 168),
    "B014": ("do_instability", 72, 96),
    "B023": ("low_cell_growth", 0, 168),
    "B031": ("excessive_aggregation", 72, 240),
    "B044": ("low_titer", 144, 240),
}


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters that define one reproducible synthetic dataset."""

    n_batches: int = 50
    duration_hr: int = 240
    sample_interval_hr: int = 4
    random_seed: int = 42
    image_size_px: int = 128
    image_timepoints_hr: tuple[int, ...] = (48, 96, 168)

    def validate(self) -> None:
        if self.n_batches < 1:
            raise ValueError("n_batches must be at least 1")
        if self.duration_hr <= 0 or self.sample_interval_hr <= 0:
            raise ValueError("duration and sample interval must be positive")
        if self.duration_hr % self.sample_interval_hr:
            raise ValueError("duration_hr must be divisible by sample_interval_hr")
        if any(
            time_hr < 0
            or time_hr > self.duration_hr
            or time_hr % self.sample_interval_hr
            for time_hr in self.image_timepoints_hr
        ):
            raise ValueError("image timepoints must fall on sampled process times")


@dataclass(frozen=True)
class GeneratedData:
    """In-memory tables produced by the simulator."""

    batches: pd.DataFrame
    sensor_data: pd.DataFrame
    outcomes: pd.DataFrame
    text_records: pd.DataFrame
    images: pd.DataFrame
    scenario_truth: pd.DataFrame


def _batch_rng(seed: int, batch_number: int) -> np.random.Generator:
    """Give each batch an independent deterministic random stream."""

    return np.random.default_rng(seed + batch_number * 10_007)


def _metadata_for_batch(
    batch_number: int, rng: np.random.Generator
) -> dict[str, object]:
    batch_id = f"B{batch_number:03d}"
    media_type = str(rng.choice(["Media_A", "Media_B", "Media_C"], p=[0.4, 0.35, 0.25]))
    feed_strategy = str(
        rng.choice(["standard", "early_feed", "intensified"], p=[0.55, 0.2, 0.25])
    )
    experiment_date = date(2024, 1, 8) + timedelta(days=7 * (batch_number - 1))
    return {
        "batch_id": batch_id,
        "cell_line": str(rng.choice(["CHO-K1", "CHO-S", "GS-CHO"])),
        "media_type": media_type,
        "media_lot": f"{media_type[-1]}-{int(rng.integers(1, 5)):02d}",
        "bioreactor_scale_l": float(rng.choice([2.0, 3.0, 5.0])),
        "seed_density_million_ml": round(float(rng.uniform(0.35, 0.65)), 4),
        "feed_strategy": feed_strategy,
        "experiment_date": experiment_date.isoformat(),
    }


def _simulate_trajectory(
    metadata: dict[str, object],
    time_hr: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    batch_id = str(metadata["batch_id"])
    scenario = SCENARIOS.get(batch_id, ("normal", None, None))[0]
    seed_density = float(metadata["seed_density_million_ml"])
    media_factor = {"Media_A": 1.0, "Media_B": 0.94, "Media_C": 1.06}[
        str(metadata["media_type"])
    ]
    feed_factor = {"standard": 1.0, "early_feed": 1.03, "intensified": 1.08}[
        str(metadata["feed_strategy"])
    ]

    carrying_capacity = 22.0 * media_factor * feed_factor * rng.normal(1.0, 0.045)
    growth_rate = float(rng.normal(0.038, 0.0025))
    if scenario == "low_cell_growth":
        carrying_capacity *= 0.58
        growth_rate *= 0.76

    exponential = np.exp(-growth_rate * time_hr)
    vcd = carrying_capacity / (
        1.0 + ((carrying_capacity / seed_density) - 1.0) * exponential
    )
    vcd *= np.exp(-float(rng.uniform(0.0045, 0.0065)) * np.maximum(time_hr - 168, 0))
    vcd += rng.normal(0, 0.18, size=time_hr.size)
    vcd = np.clip(vcd, 0.1, None)

    phase = float(rng.uniform(0, 2 * np.pi))
    do_pct = 50 + 2.2 * np.sin(2 * np.pi * time_hr / 24 + phase)
    do_pct += rng.normal(0, 1.25, size=time_hr.size)
    if scenario == "do_instability":
        event_mask = (time_hr >= 72) & (time_hr <= 96)
        do_pct[event_mask] += 14 * np.sin(2 * np.pi * (time_hr[event_mask] - 72) / 8)
        do_pct[event_mask] += rng.normal(0, 3.5, size=event_mask.sum())
    do_pct = np.clip(do_pct, 15, 85)

    feed_start = 40 if metadata["feed_strategy"] == "early_feed" else 48
    feed_multiplier = {
        "standard": 1.0,
        "early_feed": 0.95,
        "intensified": 1.2,
    }[str(metadata["feed_strategy"])]
    feed_rate = np.where(
        time_hr >= feed_start,
        feed_multiplier * (1.35 + 0.0045 * (time_hr - feed_start)),
        0.0,
    )
    feed_rate = np.clip(feed_rate + rng.normal(0, 0.035, time_hr.size), 0, None)
    cumulative_feed_ml = np.cumsum(feed_rate) * (time_hr[1] - time_hr[0])

    glucose = 5.8 - 0.018 * time_hr - 0.105 * vcd + 0.011 * cumulative_feed_ml
    glucose += 0.35 * np.sin(2 * np.pi * np.maximum(time_hr - feed_start, 0) / 24)
    glucose += rng.normal(0, 0.12, size=time_hr.size)
    glucose = np.clip(glucose, 0.25, 8.5)

    lactate = 0.12 + 0.15 * vcd
    lactate *= np.exp(-0.014 * np.maximum(time_hr - 144, 0))
    if scenario == "high_lactate":
        lactate += 2.1 / (1 + np.exp(-0.09 * (time_hr - 72)))
        lactate *= np.exp(-0.004 * np.maximum(time_hr - 168, 0))
    lactate += rng.normal(0, 0.07, size=time_hr.size)
    lactate = np.clip(lactate, 0.02, None)

    ph_noise = float(rng.uniform(0.012, 0.035))
    ph = 7.10 - 0.018 * lactate + 0.012 * np.sin(2 * np.pi * time_hr / 24 + phase)
    ph += rng.normal(0, ph_noise, size=time_hr.size)
    ph = np.clip(ph, 6.75, 7.25)

    temperature_c = 37.0 + 0.06 * np.sin(2 * np.pi * time_hr / 24 + phase)
    temperature_c += rng.normal(0, 0.035, size=time_hr.size)
    agitation_rpm = 300 + 2.4 * np.maximum(50 - do_pct, 0) + 1.15 * vcd
    agitation_rpm += rng.normal(0, 3.0, size=time_hr.size)
    air_flow_slpm = np.clip(0.45 + 0.018 * vcd + rng.normal(0, 0.018, time_hr.size), 0, None)
    o2_flow_slpm = np.clip(0.018 * np.maximum(46 - do_pct, 0), 0, None)
    co2_flow_slpm = np.clip(0.035 + 0.0075 * vcd + rng.normal(0, 0.008, time_hr.size), 0, None)

    viability_pct = 98.2 - 0.010 * np.maximum(time_hr - 120, 0) ** 1.28
    viability_pct -= 0.7 * np.maximum(lactate - 2.2, 0)
    if scenario == "excessive_aggregation":
        viability_pct -= 2.2 / (1 + np.exp(-0.08 * (time_hr - 120)))
    viability_pct += rng.normal(0, 0.25, size=time_hr.size)
    viability_pct = np.clip(viability_pct, 70, 100)

    productivity_factor = float(rng.normal(1.0, 0.055))
    if scenario == "low_titer":
        productivity_factor *= np.where(time_hr >= 144, 0.42, 1.0)
    interval_hr = float(time_hr[1] - time_hr[0])
    incremental_titer = (
        vcd
        * (viability_pct / 100)
        * productivity_factor
        * media_factor
        * interval_hr
        * 0.00095
    )
    titer_g_l = np.cumsum(incremental_titer)
    titer_g_l -= titer_g_l[0]
    titer_g_l = np.maximum.accumulate(np.clip(titer_g_l, 0, None))

    return pd.DataFrame(
        {
            "batch_id": batch_id,
            "time_hr": time_hr.astype(int),
            "temperature_c": temperature_c,
            "ph": ph,
            "do_pct": do_pct,
            "agitation_rpm": agitation_rpm,
            "air_flow_slpm": air_flow_slpm,
            "o2_flow_slpm": o2_flow_slpm,
            "co2_flow_slpm": co2_flow_slpm,
            "feed_rate_ml_hr": feed_rate,
            "glucose_g_l": glucose,
            "lactate_g_l": lactate,
            "viable_cell_density_million_ml": vcd,
            "viability_pct": viability_pct,
            "titer_g_l": titer_g_l,
        }
    )


def _notes_for_batch(
    metadata: dict[str, object], trajectory: pd.DataFrame
) -> list[dict[str, object]]:
    batch_id = str(metadata["batch_id"])
    scenario = SCENARIOS.get(batch_id, ("normal", None, None))[0]
    notes: list[tuple[int, str, str]] = [
        (
            0,
            "experiment_description",
            (
                f"Run started with {metadata['media_type']} and "
                f"{metadata['feed_strategy']} feeding."
            ),
        ),
        (48, "scientist_note", "Culture attachment and early growth appeared consistent."),
        (168, "observation", "Late-stage culture appearance documented for comparison."),
    ]
    scenario_notes = {
        "high_lactate": (
            96,
            "scientist_note",
            "Lactate remained elevated after 72 h despite continued glucose availability.",
        ),
        "do_instability": (
            84,
            "scientist_note",
            "DO became unstable after 72 h. Agitation was increased to support oxygen control.",
        ),
        "low_cell_growth": (
            120,
            "observation",
            "Cell growth was slower than historical runs after day 4.",
        ),
        "excessive_aggregation": (
            96,
            "observation",
            "Large cell aggregates were visible in the 96 h microscopy sample.",
        ),
        "low_titer": (
            192,
            "scientist_note",
            "Late-run product accumulation was below the expected trajectory.",
        ),
    }
    if scenario in scenario_notes:
        notes.append(scenario_notes[scenario])
    else:
        row_96 = trajectory.loc[trajectory["time_hr"].eq(96)].iloc[0]
        notes.append(
            (
                96,
                "observation",
                (
                    "Mid-run observation: "
                    f"VCD {row_96['viable_cell_density_million_ml']:.1f} "
                    "million cells/mL and viability "
                    f"{row_96['viability_pct']:.1f}%."
                ),
            )
        )
    return [
        {
            "batch_id": batch_id,
            "time_hr": time_hr,
            "text_type": text_type,
            "content": content,
        }
        for time_hr, text_type, content in sorted(notes)
    ]


def _draw_microscopy_image(
    output_path: Path,
    density: float,
    viability: float,
    aggregation_score: float,
    rng: np.random.Generator,
    image_size_px: int,
) -> None:
    image = Image.new("RGB", (image_size_px, image_size_px), (235, 239, 232))
    draw = ImageDraw.Draw(image, "RGBA")
    cell_count = int(np.clip(12 + density * 4.2, 18, 115))
    cluster_centers = [
        (
            int(rng.integers(18, image_size_px - 18)),
            int(rng.integers(18, image_size_px - 18)),
        )
        for _ in range(max(1, round(aggregation_score * 4)))
    ]

    for index in range(cell_count):
        if aggregation_score > 0.25 and rng.random() < aggregation_score:
            center_x, center_y = cluster_centers[index % len(cluster_centers)]
            x = int(np.clip(rng.normal(center_x, 7), 5, image_size_px - 5))
            y = int(np.clip(rng.normal(center_y, 7), 5, image_size_px - 5))
        else:
            x = int(rng.integers(5, image_size_px - 5))
            y = int(rng.integers(5, image_size_px - 5))
        radius = int(rng.integers(2, 5))
        fill = (87, 132, 112, int(rng.integers(75, 130)))
        outline = (42, 76, 64, 175)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill, outline)

    debris_probability = np.clip((96 - viability) / 20 + 0.03, 0.03, 0.55)
    debris_count = int(cell_count * debris_probability)
    for _ in range(debris_count):
        x = int(rng.integers(2, image_size_px - 2))
        y = int(rng.integers(2, image_size_px - 2))
        radius = int(rng.integers(1, 3))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(115, 91, 70, 115),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=False)


def generate_data(
    project_root: Path,
    config: SimulationConfig | None = None,
) -> GeneratedData:
    """Generate all tables and illustrative images without writing CSV files."""

    config = config or SimulationConfig()
    config.validate()
    project_root = Path(project_root).resolve()
    image_dir = project_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    time_hr = np.arange(0, config.duration_hr + 1, config.sample_interval_hr)

    batch_rows: list[dict[str, object]] = []
    sensor_frames: list[pd.DataFrame] = []
    outcome_rows: list[dict[str, object]] = []
    note_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []

    text_id = 1
    image_id = 1
    for batch_number in range(1, config.n_batches + 1):
        rng = _batch_rng(config.random_seed, batch_number)
        metadata = _metadata_for_batch(batch_number, rng)
        batch_id = str(metadata["batch_id"])
        trajectory = _simulate_trajectory(metadata, time_hr, rng)
        batch_rows.append(metadata)
        sensor_frames.append(trajectory)

        final_row = trajectory.iloc[-1]
        final_titer = float(final_row["titer_g_l"])
        final_viability = float(final_row["viability_pct"])
        max_vcd = float(trajectory["viable_cell_density_million_ml"].max())
        quality_metric = float(
            np.clip(0.65 * (final_titer / 4.5) * 100 + 0.35 * final_viability, 0, 100)
        )
        outcome_rows.append(
            {
                "batch_id": batch_id,
                "final_titer_g_l": final_titer,
                "final_viability_pct": final_viability,
                "max_vcd_million_ml": max_vcd,
                "quality_metric_score": quality_metric,
            }
        )

        for note in _notes_for_batch(metadata, trajectory):
            note_rows.append({"text_id": f"T{text_id:04d}", **note})
            text_id += 1

        scenario, event_start, event_end = SCENARIOS.get(
            batch_id, ("normal", None, None)
        )
        truth_rows.append(
            {
                "batch_id": batch_id,
                "scenario": scenario,
                "event_start_hr": event_start,
                "event_end_hr": event_end,
                "is_abnormal": scenario != "normal",
            }
        )

        for image_time in config.image_timepoints_hr:
            sensor_row = trajectory.loc[trajectory["time_hr"].eq(image_time)].iloc[0]
            aggregation = float(rng.uniform(0.05, 0.18))
            if scenario == "excessive_aggregation" and image_time >= 72:
                aggregation = 0.86
            file_name = f"{batch_id}_{image_time}h.png"
            relative_path = Path("images") / file_name
            image_rng = np.random.default_rng(
                config.random_seed + batch_number * 100_003 + image_time
            )
            _draw_microscopy_image(
                project_root / relative_path,
                density=float(sensor_row["viable_cell_density_million_ml"]),
                viability=float(sensor_row["viability_pct"]),
                aggregation_score=aggregation,
                rng=image_rng,
                image_size_px=config.image_size_px,
            )
            image_rows.append(
                {
                    "image_id": f"I{image_id:04d}",
                    "batch_id": batch_id,
                    "time_hr": image_time,
                    "file_path": relative_path.as_posix(),
                    "image_type": "synthetic_brightfield_style",
                    "is_synthetic": True,
                }
            )
            image_id += 1

    return GeneratedData(
        batches=pd.DataFrame(batch_rows),
        sensor_data=pd.concat(sensor_frames, ignore_index=True),
        outcomes=pd.DataFrame(outcome_rows),
        text_records=pd.DataFrame(note_rows),
        images=pd.DataFrame(image_rows),
        scenario_truth=pd.DataFrame(truth_rows),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    project_root: Path,
    config: SimulationConfig | None = None,
) -> GeneratedData:
    """Generate the dataset and write reproducible CSV and manifest files."""

    config = config or SimulationConfig()
    project_root = Path(project_root).resolve()
    raw_dir = project_root / "data" / "raw"
    evaluation_dir = project_root / "data" / "evaluation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_data(project_root, config)

    table_paths = {
        "batches": raw_dir / "batches.csv",
        "sensor_data": raw_dir / "sensor_data.csv",
        "outcomes": raw_dir / "outcomes.csv",
        "text_records": raw_dir / "text_records.csv",
        "images": raw_dir / "images.csv",
        "scenario_truth": evaluation_dir / "scenario_truth.csv",
    }
    for table_name, output_path in table_paths.items():
        getattr(dataset, table_name).to_csv(
            output_path,
            index=False,
            float_format="%.6f",
            lineterminator="\n",
        )

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "config": asdict(config),
        "notice": (
            "Synthetic educational bioprocess data. Microscopy-style images are "
            "illustrative and are not real microscopy measurements."
        ),
        "assumptions": [
            "Generic mammalian fed-batch behavior represented by illustrative equations.",
            "Titer is derived from integrated viable biomass and a synthetic productivity factor.",
            "Notes and images are generated from the same simulated process state.",
        ],
        "units": {
            "time_hr": "hour",
            "temperature_c": "degree Celsius",
            "ph": "dimensionless",
            "do_pct": "percent air saturation",
            "agitation_rpm": "revolutions per minute",
            "air_flow_slpm": "standard liter per minute",
            "o2_flow_slpm": "standard liter per minute",
            "co2_flow_slpm": "standard liter per minute",
            "feed_rate_ml_hr": "milliliter per hour",
            "glucose_g_l": "gram per liter",
            "lactate_g_l": "gram per liter",
            "viable_cell_density_million_ml": "million cells per milliliter",
            "viability_pct": "percent",
            "titer_g_l": "gram per liter",
        },
        "row_counts": {
            table_name: len(getattr(dataset, table_name))
            for table_name in table_paths
        },
        "files": {
            str(path.relative_to(project_root).as_posix()): _file_sha256(path)
            for path in table_paths.values()
        },
    }
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root where data/ and images/ will be written.",
    )
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = SimulationConfig(n_batches=args.batches, random_seed=args.seed)
    dataset = write_dataset(args.project_root, config)
    print(
        "Generated "
        f"{len(dataset.batches)} batches, "
        f"{len(dataset.sensor_data)} sensor rows, and "
        f"{len(dataset.images)} illustrative images."
    )


if __name__ == "__main__":
    main()
