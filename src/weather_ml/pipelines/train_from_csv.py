from __future__ import annotations

import argparse
import logging

import pandas as pd

from weather_ml.training import XGBoostTrainingParams, train_weather_condition_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    csv_path: str = "exports/openmeteo_hourly_features_istanbul_2010_2026_supabase_import_clean.csv",
    output_dir: str = "artifacts/training",
    max_rows: int | None = None,
    log_to_mlflow: bool = True,
    test_fraction: float = 0.2,
    training_params: XGBoostTrainingParams | None = None,
) -> None:
    frame = pd.read_csv(csv_path)
    if max_rows is not None:
        frame = frame.head(max_rows)
    logger.info("read feature csv: csv=%s rows=%s", csv_path, len(frame))
    artifacts = train_weather_condition_model(
        frame,
        output_dir=output_dir,
        test_fraction=test_fraction,
        log_to_mlflow=log_to_mlflow,
        training_params=training_params,
    )
    logger.info("trained model: model=%s metrics=%s predictions=%s", artifacts.model_path, artifacts.metrics_path, artifacts.predictions_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="exports/openmeteo_hourly_features_istanbul_2010_2026_supabase_import_clean.csv")
    parser.add_argument("--output-dir", default="artifacts/training")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
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
        training_params=XGBoostTrainingParams(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            acceptance_threshold=args.acceptance_threshold,
        ),
    )


if __name__ == "__main__":
    main()
