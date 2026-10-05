"""Train the Milestone 10 multimodal ablation experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.models.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.models.multimodal import PROJECT_ROOT, build_multimodal_feature_table

TABULAR_MANIFEST = PROJECT_ROOT / "artifacts" / "tabular" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "multimodal"
RANDOM_SEED = 42
PCA_COMPONENTS = 20
FEATURE_SETS = {
    "Tabular only": (),
    "Tabular + text": ("text",),
    "Tabular + image": ("image",),
    "Tabular + text + image": ("text", "image"),
}


def _embedding_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [
        column
        for column in frame
        if column.startswith(f"{prefix}_") and column[-3:].isdigit()
    ]


def build_preprocessor(frame: pd.DataFrame, modalities: tuple[str, ...]) -> ColumnTransformer:
    numeric = NUMERIC_FEATURES + [
        feature
        for modality in modalities
        for feature in (f"{modality}_count", f"{modality}_available")
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
    for modality in modalities:
        columns = _embedding_columns(frame, modality)
        transformers.append(
            (
                modality,
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)),
                    ]
                ),
                columns,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _estimators() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "PLS": PLSRegression(n_components=5, scale=False),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
    }


def train_and_save(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    frame = build_multimodal_feature_table()
    tabular_manifest = json.loads(TABULAR_MANIFEST.read_text(encoding="utf-8"))
    train_ids = tabular_manifest["split"]["train_batch_ids"]
    test_ids = tabular_manifest["split"]["test_batch_ids"]
    train = frame.loc[frame["batch_id"].isin(train_ids)].copy()
    test = frame.loc[frame["batch_id"].isin(test_ids)].copy()
    threshold = float(train[TARGET].quantile(0.25))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []

    for feature_set, modalities in FEATURE_SETS.items():
        for model_name, estimator in _estimators().items():
            pipeline = Pipeline(
                [("preprocessor", build_preprocessor(frame, modalities)), ("model", estimator)]
            )
            started = time.perf_counter()
            scores = cross_validate(
                pipeline,
                train.drop(columns=[TARGET, "batch_id"]),
                train[TARGET],
                cv=cv,
                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                n_jobs=1,
            )
            pipeline.fit(train.drop(columns=[TARGET, "batch_id"]), train[TARGET])
            predicted = np.asarray(
                pipeline.predict(test.drop(columns=[TARGET, "batch_id"]))
            ).reshape(-1)
            actual_low = test[TARGET].to_numpy() < threshold
            predicted_low = predicted < threshold
            matrix = confusion_matrix(actual_low, predicted_low, labels=[False, True])
            transformed_count = pipeline.named_steps["preprocessor"].transform(
                train.drop(columns=[TARGET, "batch_id"]).iloc[:1]
            ).shape[1]
            row = {
                "feature_set": feature_set,
                "model": model_name,
                "cv_rmse_mean": float(-scores["test_rmse"].mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "test_rmse": float(mean_squared_error(test[TARGET], predicted) ** 0.5),
                "test_mae": float(mean_absolute_error(test[TARGET], predicted)),
                "test_r2": float(r2_score(test[TARGET], predicted)),
                "low_titer_precision": float(precision_score(actual_low, predicted_low, zero_division=0)),
                "low_titer_recall": float(recall_score(actual_low, predicted_low, zero_division=0)),
                "low_titer_f1": float(f1_score(actual_low, predicted_low, zero_division=0)),
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
                "feature_count": transformed_count,
                "text_pca_components": PCA_COMPONENTS if "text" in modalities else 0,
                "image_pca_components": PCA_COMPONENTS if "image" in modalities else 0,
                "training_seconds": time.perf_counter() - started,
            }
            results.append(row)
            predictions.append(
                pd.DataFrame(
                    {
                        "batch_id": test["batch_id"].to_numpy(),
                        "feature_set": feature_set,
                        "model": model_name,
                        "actual_titer_g_l": test[TARGET].to_numpy(),
                        "predicted_titer_g_l": predicted,
                        "residual_g_l": test[TARGET].to_numpy() - predicted,
                        "actual_low_titer": actual_low,
                        "predicted_low_titer": predicted_low,
                    }
                )
            )
            if feature_set == "Tabular + text + image":
                model_id = model_name.lower().replace(" ", "_")
                joblib.dump(pipeline, output_dir / f"{model_id}_multimodal-v1.joblib")

    result_frame = pd.DataFrame(results).sort_values(["test_rmse", "feature_set"])
    prediction_frame = pd.concat(predictions, ignore_index=True)
    frame.to_csv(output_dir / "fused_batch_features.csv", index=False)
    result_frame.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    manifest = {
        "model_version": "multimodal-v1",
        "task": "end-of-run final titer regression with multimodal ablations",
        "batch_count": len(frame),
        "train_batch_ids": train_ids,
        "test_batch_ids": test_ids,
        "low_titer_threshold_g_l": threshold,
        "pca": {
            "requested_components": PCA_COMPONENTS,
            "fit_scope": "inside each training/CV pipeline",
            "text_input_dimension": 384,
            "image_input_dimension": 512,
        },
        "feature_sets": list(FEATURE_SETS),
        "limitations": [
            "Only 50 synthetic batches are available for a high-dimensional experiment.",
            "Text and images derive from the same simulator state and are not independent evidence.",
            "All modalities use data through 240 hours, so this is an end-of-run experiment.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = train_and_save(args.output_dir)
    print(f"Completed multimodal ablations for {manifest['batch_count']} batches.")


if __name__ == "__main__":
    main()
