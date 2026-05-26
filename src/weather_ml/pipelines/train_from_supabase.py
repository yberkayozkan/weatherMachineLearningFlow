from __future__ import annotations

import argparse
import logging

from weather_ml.config import get_settings, require_supabase_rest_credentials
from weather_ml.supabase_rest import classify_supabase_key, read_table_rest
from weather_ml.training import XGBoostTrainingParams, train_weather_condition_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    table_name: str = "KadikoyWeatherCodeFeature",
    output_dir: str = "artifacts/training",
    max_rows: int | None = None,
    log_to_mlflow: bool = True,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    training_params: XGBoostTrainingParams | None = None,
) -> None:
    supabase_url, supabase_key = require_supabase_rest_credentials()
    key_type = classify_supabase_key(supabase_key)
    settings = get_settings()
    if settings.supabase_service_role_key and key_type == "publishable":
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY contains a publishable key. "
            "Training reads Supabase through REST and needs the real service_role/secret key "
            "so RLS does not hide rows. Put the service role key in the GitHub secret "
            "SUPABASE_SERVICE_ROLE_KEY, not SUPABASE_KEY."
        )
    if not settings.supabase_service_role_key:
        logger.warning(
            "SUPABASE_SERVICE_ROLE_KEY is not set; falling back to SUPABASE_KEY. "
            "If RLS is enabled, the feature table may appear empty."
        )
    logger.info("using Supabase REST credentials: url=%s key_type=%s", supabase_url, key_type)
    frame = read_table_rest(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=table_name,
        order_by="observed_at",
        max_rows=max_rows,
    )
    logger.info("read feature table from Supabase: table=%s rows=%s", table_name, len(frame))
    if frame.empty:
        raise RuntimeError(
            f"Supabase feature table is empty: {table_name}. "
            "Create the table with sql/005_kadikoy_weather_code_tables.sql, then import "
            "cache/kadikoy_weather_condition_features_full_import.csv or run the "
            "OpenMeteo Supabase Pipeline before training."
        )
    artifacts = train_weather_condition_model(
        frame,
        output_dir=output_dir,
        test_fraction=test_fraction,
        cv_splits=cv_splits,
        log_to_mlflow=log_to_mlflow,
        training_params=training_params,
    )
    logger.info(
        "trained model: model=%s metrics=%s predictions=%s",
        artifacts.model_path,
        artifacts.metrics_path,
        artifacts.predictions_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-name", default="KadikoyWeatherCodeFeature")
    parser.add_argument("--output-dir", default="artifacts/training")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--acceptance-threshold", type=float, default=0.40)
    args = parser.parse_args()
    run(
        table_name=args.table_name,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        log_to_mlflow=not args.no_mlflow,
        test_fraction=args.test_fraction,
        cv_splits=args.cv_splits,
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
