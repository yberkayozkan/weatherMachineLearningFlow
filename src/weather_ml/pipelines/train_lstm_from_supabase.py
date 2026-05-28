from __future__ import annotations

import argparse
import logging

import pandas as pd

from weather_ml.config import get_settings, require_supabase_rest_credentials
from weather_ml.lstm_training import LSTMTrainingParams, train_lstm_weather_condition_model
from weather_ml.supabase_rest import classify_supabase_key, read_table_rest
from weather_ml.training import HISTORICAL_FORECAST_TRAINING_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    table_name: str = "KadikoyWeatherCodeFeature",
    output_dir: str = "artifacts/lstm-training",
    max_rows: int | None = None,
    log_to_mlflow: bool = True,
    test_fraction: float = 0.2,
    training_params: LSTMTrainingParams | None = None,
) -> None:
    supabase_url, supabase_key = require_supabase_rest_credentials()
    settings = get_settings()
    key_type = classify_supabase_key(supabase_key)
    if settings.supabase_service_role_key and key_type == "publishable":
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY contains a publishable key.")
    if not settings.supabase_service_role_key:
        logger.warning("SUPABASE_SERVICE_ROLE_KEY is not set; falling back to SUPABASE_KEY.")
    frame = read_table_rest(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=table_name,
        order_by="observed_at",
    )
    if frame.empty:
        raise RuntimeError(f"Supabase feature table is empty: {table_name}.")
    timestamps = pd.to_datetime(frame["observed_at"], errors="coerce")
    frame = frame.loc[timestamps >= pd.Timestamp(HISTORICAL_FORECAST_TRAINING_START)].copy()
    if max_rows is not None:
        frame = frame.tail(max_rows)
    artifacts = train_lstm_weather_condition_model(
        frame,
        output_dir=output_dir,
        test_fraction=test_fraction,
        log_to_mlflow=log_to_mlflow,
        training_params=training_params,
    )
    logger.info(
        "trained LSTM model: model=%s metrics=%s",
        artifacts.model_path,
        artifacts.metrics_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-name", default="KadikoyWeatherCodeFeature")
    parser.add_argument("--output-dir", default="artifacts/lstm-training")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--acceptance-threshold", type=float, default=0.40)
    args = parser.parse_args()
    run(
        table_name=args.table_name,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        log_to_mlflow=not args.no_mlflow,
        test_fraction=args.test_fraction,
        training_params=LSTMTrainingParams(
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            acceptance_threshold=args.acceptance_threshold,
        ),
    )


if __name__ == "__main__":
    main()
