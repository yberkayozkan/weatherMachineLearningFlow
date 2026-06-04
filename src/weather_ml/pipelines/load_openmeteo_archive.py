from __future__ import annotations

import argparse
import logging

import pandas as pd

from weather_ml.config import (
    get_settings,
    require_supabase_db_url,
    require_supabase_rest_credentials,
)
from weather_ml.supabase_db import assert_connection, make_engine, upsert_dataframe
from weather_ml.supabase_rest import upsert_dataframe_rest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    csv_path: str = "exports/openmeteo_hourly_istanbul_2010_2026_supabase_import_clean.csv",
    table_name: str = "KadikoyWeatherCodeRaw",
    conflict_column: str = "observed_at",
    method: str = "auto",
) -> None:
    frame = pd.read_csv(csv_path)
    settings = get_settings()
    if method not in {"auto", "postgres", "rest"}:
        raise ValueError("method must be one of: auto, postgres, rest")

    if method == "postgres" or (method == "auto" and settings.supabase_db_url):
        engine = make_engine(require_supabase_db_url())
        assert_connection(engine)
        rows = upsert_dataframe(
            engine,
            frame,
            table_name=table_name,
            conflict_columns=[conflict_column],
        )
    else:
        supabase_url, supabase_key = require_supabase_rest_credentials()
        rows = upsert_dataframe_rest(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            table_name=table_name,
            frame=frame,
            on_conflict=conflict_column,
        )
    logger.info("loaded openmeteo archive: rows=%s table=%s csv=%s", rows, table_name, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        default="exports/openmeteo_hourly_istanbul_latest_supabase_import.csv",
    )
    parser.add_argument("--table-name", default="KadikoyWeatherCodeRaw")
    parser.add_argument("--conflict-column", default="observed_at")
    parser.add_argument("--method", choices=["auto", "postgres", "rest"], default="auto")
    args = parser.parse_args()
    run(
        csv_path=args.csv_path,
        table_name=args.table_name,
        conflict_column=args.conflict_column,
        method=args.method,
    )


if __name__ == "__main__":
    main()
