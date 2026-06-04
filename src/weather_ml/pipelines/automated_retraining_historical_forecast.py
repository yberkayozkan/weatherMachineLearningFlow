from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from weather_ml.config import get_settings, require_supabase_rest_credentials
from weather_ml.drift import build_drift_report
from weather_ml.model_registry import (
    PRODUCTION_ALIAS,
    build_promotion_decision,
    configure_mlflow_tracking_from_env,
    get_latest_model_version,
    get_production_model_snapshot,
    promote_model_version,
    write_promotion_decision,
)
from weather_ml.supabase_rest import classify_supabase_key, read_table_rest
from weather_ml.training import (
    HISTORICAL_FORECAST_FEATURE_COLUMNS,
    HISTORICAL_FORECAST_TRAINING_START,
    HISTORICAL_FORECAST_XGBOOST_REGISTERED_MODEL_NAME,
    XGBoostTrainingParams,
    train_historical_forecast_xgboost_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    table_name: str = "KadikoyWeatherCodeFeature",
    output_dir: str = "artifacts/automated-historical-forecast-retraining",
    current_window_days: int = 30,
    promotion_metric: str = "f1_macro",
    min_improvement: float = 0.02,
    psi_warning_threshold: float = 0.10,
    psi_drift_threshold: float = 0.20,
    promote: bool = True,
) -> None:
    configure_mlflow_tracking_from_env()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = _read_feature_table(table_name)
    eligible = _eligible_historical_forecast_rows(frame)
    current = _current_window(eligible, current_window_days=current_window_days)

    production_snapshot = get_production_model_snapshot(
        model_name=HISTORICAL_FORECAST_XGBOOST_REGISTERED_MODEL_NAME,
        artifact_dir=out / "production",
        alias=PRODUCTION_ALIAS,
    )
    drift_report = build_drift_report(
        baseline=None if production_snapshot is None else production_snapshot.baseline,
        current_frame=current,
        feature_columns=HISTORICAL_FORECAST_FEATURE_COLUMNS,
        warning_threshold=psi_warning_threshold,
        drift_threshold=psi_drift_threshold,
    )
    drift_report.update(
        {
            "current_window_days": current_window_days,
            "production_model": None
            if production_snapshot is None
            else {
                "name": production_snapshot.model_name,
                "version": production_snapshot.version,
                "alias": production_snapshot.alias,
                "run_id": production_snapshot.run_id,
            },
        }
    )
    drift_report_path = out / "drift_report.json"
    drift_report_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

    training_output_dir = out / "training"
    artifacts = train_historical_forecast_xgboost_model(
        eligible,
        output_dir=str(training_output_dir),
        log_to_mlflow=True,
        training_params=XGBoostTrainingParams(class_weight_power=1.0),
    )
    candidate_metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    candidate_version = get_latest_model_version(
        model_name=HISTORICAL_FORECAST_XGBOOST_REGISTERED_MODEL_NAME
    )
    decision = build_promotion_decision(
        candidate_metrics=candidate_metrics,
        candidate_version=candidate_version,
        production_snapshot=production_snapshot,
        promotion_metric=promotion_metric,
        min_improvement=min_improvement,
        alias=PRODUCTION_ALIAS,
    )
    if promote and decision.should_promote and decision.candidate_version is not None:
        promote_model_version(
            model_name=HISTORICAL_FORECAST_XGBOOST_REGISTERED_MODEL_NAME,
            version=decision.candidate_version,
            alias=PRODUCTION_ALIAS,
        )
        logger.info(
            "promoted candidate model: name=%s version=%s alias=%s",
            HISTORICAL_FORECAST_XGBOOST_REGISTERED_MODEL_NAME,
            decision.candidate_version,
            PRODUCTION_ALIAS,
        )
    elif not promote and decision.should_promote:
        decision = decision.__class__(
            should_promote=False,
            reason=f"dry_run_{decision.reason}",
            candidate_version=decision.candidate_version,
            production_version=decision.production_version,
            candidate_metric=decision.candidate_metric,
            production_metric=decision.production_metric,
            min_improvement=decision.min_improvement,
            promoted_alias=decision.promoted_alias,
        )

    write_promotion_decision(out / "promotion_decision.json", decision)
    logger.info(
        "automated retraining complete: drift=%s decision=%s output=%s",
        drift_report["status"],
        decision.reason,
        out,
    )


def _read_feature_table(table_name: str) -> pd.DataFrame:
    supabase_url, supabase_key = require_supabase_rest_credentials()
    key_type = classify_supabase_key(supabase_key)
    settings = get_settings()
    if settings.supabase_service_role_key and key_type == "publishable":
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY contains a publishable key. "
            "Automated retraining needs the service_role/secret key for REST reads."
        )
    frame = read_table_rest(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=table_name,
        order_by="observed_at",
    )
    if frame.empty:
        raise RuntimeError(f"Supabase feature table is empty: {table_name}.")
    return frame


def _eligible_historical_forecast_rows(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["observed_at"], errors="coerce")
    eligible = frame.loc[timestamps >= pd.Timestamp(HISTORICAL_FORECAST_TRAINING_START)].copy()
    if eligible.empty:
        raise RuntimeError(
            "No rows are available on or after "
            f"{HISTORICAL_FORECAST_TRAINING_START} for automated retraining."
        )
    return eligible


def _current_window(frame: pd.DataFrame, *, current_window_days: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["observed_at"], errors="coerce")
    max_timestamp = timestamps.max()
    if pd.isna(max_timestamp):
        return frame.iloc[0:0].copy()
    start = max_timestamp - pd.Timedelta(days=current_window_days)
    return frame.loc[timestamps >= start].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-name", default="KadikoyWeatherCodeFeature")
    parser.add_argument(
        "--output-dir",
        default="artifacts/automated-historical-forecast-retraining",
    )
    parser.add_argument("--current-window-days", type=int, default=30)
    parser.add_argument("--promotion-metric", default="f1_macro")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--psi-warning-threshold", type=float, default=0.10)
    parser.add_argument("--psi-drift-threshold", type=float, default=0.20)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    run(
        table_name=args.table_name,
        output_dir=args.output_dir,
        current_window_days=args.current_window_days,
        promotion_metric=args.promotion_metric,
        min_improvement=args.min_improvement,
        psi_warning_threshold=args.psi_warning_threshold,
        psi_drift_threshold=args.psi_drift_threshold,
        promote=not args.no_promote,
    )


if __name__ == "__main__":
    main()
