from __future__ import annotations

import argparse
import logging

import pandas as pd

from weather_ml.lightgbm_training import (
    LightGBMTrainingParams,
    train_lightgbm_weather_condition_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    csv_path: str = (
        "exports/openmeteo_hourly_features_istanbul_2010_2026_supabase_import_clean.csv"
    ),
    output_dir: str = "artifacts/lightgbm-training",
    max_rows: int | None = None,
    log_to_mlflow: bool = True,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    training_params: LightGBMTrainingParams | None = None,
) -> None:
    frame = pd.read_csv(csv_path)
    if max_rows is not None:
        frame = frame.head(max_rows)
    artifacts = train_lightgbm_weather_condition_model(
        frame,
        output_dir=output_dir,
        test_fraction=test_fraction,
        cv_splits=cv_splits,
        log_to_mlflow=log_to_mlflow,
        training_params=training_params,
    )
    logger.info(
        "trained LightGBM model: model=%s metrics=%s",
        artifacts.model_path,
        artifacts.metrics_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/lightgbm-training")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--acceptance-threshold", type=float, default=0.40)
    args = parser.parse_args()
    run(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        log_to_mlflow=not args.no_mlflow,
        test_fraction=args.test_fraction,
        cv_splits=args.cv_splits,
        training_params=LightGBMTrainingParams(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            acceptance_threshold=args.acceptance_threshold,
        ),
    )


if __name__ == "__main__":
    main()
