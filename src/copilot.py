"""Grounded evidence construction and OpenRouter explanation client."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from src.data.repository import (
    load_batch_catalog,
    load_sensor_data,
    load_text_records,
    select_reference_batches,
)
from src.embeddings.store import search_similar_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_local_env(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_evidence_bundle(batch_id: str) -> dict:
    catalog = load_batch_catalog()
    references = select_reference_batches(catalog, batch_id, "Same media")
    selected_sensor = load_sensor_data([batch_id])
    reference_sensor = load_sensor_data(references)
    selected_outcome = catalog.loc[catalog["batch_id"].eq(batch_id)].iloc[0]
    reference_outcomes = catalog.loc[catalog["batch_id"].isin(references)]

    selected_summary = {
        "do_std_pct": float(selected_sensor["do_pct"].std()),
        "peak_vcd_million_ml": float(selected_sensor["viable_cell_density_million_ml"].max()),
        "max_lactate_g_l": float(selected_sensor["lactate_g_l"].max()),
    }
    per_batch = reference_sensor.groupby("batch_id").agg(
        do_std_pct=("do_pct", "std"),
        peak_vcd_million_ml=("viable_cell_density_million_ml", "max"),
        max_lactate_g_l=("lactate_g_l", "max"),
    )
    comparison = {}
    for metric, value in selected_summary.items():
        reference_mean = float(per_batch[metric].mean())
        comparison[metric] = {
            "selected": value,
            "reference_mean": reference_mean,
            "difference_pct": 100 * (value - reference_mean) / reference_mean,
        }

    notes = load_text_records(batch_id)
    image_id = f"I{((int(batch_id[1:]) - 1) * 3 + 2):04d}"
    similar_images = search_similar_images(image_id, limit=3)
    return {
        "batch_id": batch_id,
        "data_provenance": "Synthetic educational simulator; notes and images share simulator state.",
        "outcome": {
            "final_titer_g_l": float(selected_outcome["final_titer_g_l"]),
            "reference_mean_titer_g_l": float(reference_outcomes["final_titer_g_l"].mean()),
            "reference_std_titer_g_l": float(reference_outcomes["final_titer_g_l"].std()),
            "final_viability_pct": float(selected_outcome["final_viability_pct"]),
        },
        "sensor_comparison": comparison,
        "notes": notes[["time_hr", "text_type", "content"]].to_dict("records"),
        "similar_images": [
            {
                "batch_id": item["batch_id"],
                "time_hr": item["time_hr"],
                "similarity": item["similarity"],
            }
            for item in similar_images
        ],
    }


def build_prompt(question: str, evidence: dict) -> list[dict[str, str]]:
    system = (
        "You are a bioprocess data-analysis copilot. Use only the supplied evidence. "
        "State measured observations first, label causal ideas as hypotheses, and mention "
        "that all evidence is synthetic and that notes/images share the simulator state. "
        "Do not invent values or claim independent confirmation."
    )
    user = f"Question: {question}\n\nEvidence JSON:\n{json.dumps(evidence, indent=2)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ask_openrouter(question: str, evidence: dict, timeout: float = 60) -> dict:
    load_local_env()
    api_key = os.getenv("OPEN_ROUTER_API") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured")
    configured_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    models = [configured_model]
    if configured_model != "openrouter/free":
        models.append("openrouter/free")
    last_error: requests.RequestException | None = None
    for model in models:
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/bioprocess-multimodal-intelligence",
                    "X-OpenRouter-Title": "Bioprocess Multimodal Intelligence",
                },
                json={
                    "model": model,
                    "messages": build_prompt(question, evidence),
                    "temperature": 0.2,
                    "max_tokens": 1_800,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload["choices"][0]["message"]
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("OpenRouter returned an empty explanation")
            return {
                "answer": content,
                "requested_model": model,
                "served_model": payload.get("model", model),
                "usage": payload.get("usage", {}),
            }
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, requests.RequestException):
                last_error = error
            if model == models[-1]:
                raise RuntimeError(f"OpenRouter request failed: {error}") from error
    raise RuntimeError(f"OpenRouter request failed: {last_error}")
