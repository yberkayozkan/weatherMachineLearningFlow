from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

from weather_ml.pipelines.build_openmeteo_features import run as build_features
from weather_ml.pipelines.download_openmeteo_archive import run as download_archive
from weather_ml.pipelines.load_openmeteo_archive import run as load_archive
from weather_ml.pipelines.load_openmeteo_features import run as load_features
from weather_ml.openmeteo_features import map_weather_code_to_condition


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_csv: str = "data/openmeteo_archive/Istanbul/latest/openmeteo_hourly.csv",
    raw_upload_csv: str = "exports/openmeteo_hourly_istanbul_latest_supabase_import.csv",
    feature_context_csv: str = "exports/openmeteo_hourly_istanbul_latest_feature_context.csv",
    feature_csv: str = "exports/openmeteo_hourly_features_istanbul_latest_supabase_import.csv",
    feature_summary_json: str = "exports/openmeteo_hourly_features_summary.json",
    method: str = "rest",
    raw_table: str = "KadikoyWeatherCodeRaw",
    feature_table: str = "KadikoyWeatherCodeFeature",
    raw_conflict_column: str = "observed_at",
    feature_conflict_column: str = "observed_at",
    max_persist_observed_at: str | None = None,
    push_raw: bool = True,
    push_features: bool = True,
) -> None:
    download_archive(start_date=start_date, end_date=end_date)
    persist_cutoff = max_persist_observed_at or _default_persist_cutoff(end_date)
    _prepare_raw_upload_csv(raw_csv, raw_upload_csv, max_observed_at=persist_cutoff)
    _prepare_raw_upload_csv(raw_csv, feature_context_csv)
    build_features(
        input_csv=feature_context_csv,
        output_csv=feature_csv,
        summary_json=feature_summary_json,
        max_observed_at=persist_cutoff,
    )

    if push_raw:
        load_archive(csv_path=raw_upload_csv, table_name=raw_table, conflict_column=raw_conflict_column, method=method)
    if push_features:
        load_features(csv_path=feature_csv, table_name=feature_table, conflict_column=feature_conflict_column, method=method)


def _prepare_raw_upload_csv(input_csv: str, output_csv: str, max_observed_at: str | None = None) -> None:
    import pandas as pd

    frame = pd.read_csv(input_csv)
    frame = frame.drop(columns=[column for column in ["source", "location_name"] if column in frame.columns])
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    if max_observed_at:
        cutoff = pd.to_datetime(max_observed_at)
        observed_for_filter = pd.to_datetime(frame["observed_at"])
        frame = frame[observed_for_filter <= cutoff].copy().reset_index(drop=True)
    observed = pd.to_datetime(frame["observed_at"], utc=False)
    base = pd.Timestamp(date(2010, 1, 1))
    frame.insert(0, "id", ((observed - base) / pd.Timedelta(hours=1)).round().astype("int64") + 1)
    if "weather_code" in frame.columns:
        frame["weather_code"] = pd.to_numeric(frame["weather_code"], errors="coerce").round().astype("Int64")
        frame["weather_condition"] = frame["weather_code"].map(map_weather_code_to_condition)
    frame = frame.drop_duplicates(subset=["observed_at"], keep="last").reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    logger.info(
        "prepared raw upload csv: rows=%s max_observed_at=%s output=%s",
        len(frame),
        max_observed_at,
        output_csv,
    )


def _default_persist_cutoff(end_date: str | None) -> str | None:
    if not end_date:
        return None

    cutoff_date = date.fromisoformat(end_date) - timedelta(days=1)
    return f"{cutoff_date.isoformat()} 23:00:00"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--raw-csv", default="data/openmeteo_archive/Istanbul/latest/openmeteo_hourly.csv")
    parser.add_argument("--raw-upload-csv", default="exports/openmeteo_hourly_istanbul_latest_supabase_import.csv")
    parser.add_argument("--feature-context-csv", default="exports/openmeteo_hourly_istanbul_latest_feature_context.csv")
    parser.add_argument("--feature-csv", default="exports/openmeteo_hourly_features_istanbul_latest_supabase_import.csv")
    parser.add_argument("--feature-summary-json", default="exports/openmeteo_hourly_features_summary.json")
    parser.add_argument("--method", choices=["postgres", "rest"], default="rest")
    parser.add_argument("--raw-table", default="KadikoyWeatherCodeRaw")
    parser.add_argument("--feature-table", default="KadikoyWeatherCodeFeature")
    parser.add_argument("--raw-conflict-column", default="observed_at")
    parser.add_argument("--feature-conflict-column", default="observed_at")
    parser.add_argument("--max-persist-observed-at")
    parser.add_argument("--max-feature-observed-at", dest="max_feature_observed_at")
    parser.add_argument("--skip-raw-push", action="store_true")
    parser.add_argument("--skip-feature-push", action="store_true")
    args = parser.parse_args()
    run(
        start_date=args.start_date,
        end_date=args.end_date,
        raw_csv=args.raw_csv,
        raw_upload_csv=args.raw_upload_csv,
        feature_context_csv=args.feature_context_csv,
        feature_csv=args.feature_csv,
        feature_summary_json=args.feature_summary_json,
        method=args.method,
        raw_table=args.raw_table,
        feature_table=args.feature_table,
        raw_conflict_column=args.raw_conflict_column,
        feature_conflict_column=args.feature_conflict_column,
        max_persist_observed_at=args.max_persist_observed_at or args.max_feature_observed_at,
        push_raw=not args.skip_raw_push,
        push_features=not args.skip_feature_push,
    )


if __name__ == "__main__":
    main()
