"""Train and evaluate Milestone 5 tabular final-titer models."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data.repository import load_batches, load_outcomes, load_sensor_data
from src.models.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    engineer_batch_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tabular"
MODEL_VERSION = "tabular-v1"
RANDOM_SEED = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    estimators = {
        "Linear Regression": LinearRegression(),
        "PLS": PLSRegression(n_components=5, scale=False),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=1,
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
    return {
        name: Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        values = np.abs(coefficients.reshape(-1))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(values) != len(names):
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_save(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    features = engineer_batch_features(
        load_batches(), load_sensor_data(), load_outcomes()
    )
    train_ids, test_ids = train_test_split(
        features["batch_id"].to_numpy(),
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train = features.loc[features["batch_id"].isin(train_ids)].copy()
    test = features.loc[features["batch_id"].isin(test_ids)].copy()
    x_train, y_train = train[MODEL_FEATURES], train[TARGET]
    x_test, y_test = test[MODEL_FEATURES], test[TARGET]
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for model_name, pipeline in build_model_pipelines().items():
        started = time.perf_counter()
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cross_validation,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        predicted = np.asarray(pipeline.predict(x_test)).reshape(-1)
        elapsed = time.perf_counter() - started
        model_id = model_name.lower().replace(" ", "_")
        artifact_name = f"{model_id}_{MODEL_VERSION}.joblib"
        joblib.dump(pipeline, output_dir / artifact_name)

        results.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(-scores["test_rmse"].mean()),
                "cv_rmse_std": float(scores["test_rmse"].std()),
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "test_rmse": float(mean_squared_error(y_test, predicted) ** 0.5),
                "test_mae": float(mean_absolute_error(y_test, predicted)),
                "test_r2": float(r2_score(y_test, predicted)),
                "training_seconds": elapsed,
                "artifact": artifact_name,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "batch_id": test["batch_id"].to_numpy(),
                    "model": model_name,
                    "actual_titer_g_l": y_test.to_numpy(),
                    "predicted_titer_g_l": predicted,
                    "residual_g_l": y_test.to_numpy() - predicted,
                }
            )
        )
        importance = _feature_importance(pipeline).head(15)
        importance.insert(0, "model", model_name)
        importance_frames.append(importance)

    results_frame = pd.DataFrame(results).sort_values("test_rmse")
    predictions_frame = pd.concat(predictions, ignore_index=True)
    importance_frame = pd.concat(importance_frames, ignore_index=True)
    features.to_csv(output_dir / "batch_features.csv", index=False)
    results_frame.to_csv(output_dir / "metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)

    manifest = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "task": "end-of-run final titer regression",
        "target": {"name": TARGET, "unit": "g/L"},
        "feature_cutoff_hr": 240,
        "feature_columns": MODEL_FEATURES,
        "excluded": [
            "batch_id",
            "experiment_date",
            "media_lot",
            "all titer trajectory values",
            "planted scenario labels",
        ],
        "split": {
            "strategy": "batch-level fixed holdout",
            "random_seed": RANDOM_SEED,
            "train_batch_ids": sorted(train_ids.tolist()),
            "test_batch_ids": sorted(test_ids.tolist()),
            "cross_validation": "5-fold shuffled KFold on training batches only",
        },
        "data": {"batch_count": len(features), "training_count": len(train), "test_count": len(test)},
        "libraries": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "models": results_frame.to_dict(orient="records"),
        "limitations": [
            "Only 50 synthetic batches are available, so held-out metrics are uncertain.",
            "Features use measurements through 240 hours and cannot be interpreted as an early forecast.",
            "Synthetic relationships may be easier to learn than real bioprocess behavior.",
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
    print(
        f"Trained {len(manifest['models'])} models on "
        f"{manifest['data']['training_count']} batches; artifacts: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
