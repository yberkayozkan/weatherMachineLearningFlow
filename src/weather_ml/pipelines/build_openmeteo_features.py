from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from weather_ml.openmeteo_features import (
    build_openmeteo_hourly_features,
    clean_features_for_training,
    validate_feature_upload_frame,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    input_csv: str = "exports/openmeteo_hourly_istanbul_2010_2026_supabase_import_clean.csv",
    output_csv: str = "exports/openmeteo_hourly_features_istanbul_2010_2026_supabase_import_clean.csv",
    summary_json: str = "exports/openmeteo_hourly_features_summary.json",
    keep_incomplete_rows: bool = False,
    max_observed_at: str | None = None,
) -> None:
    raw = pd.read_csv(input_csv)
    features = build_openmeteo_hourly_features(raw)
    output = features if keep_incomplete_rows else clean_features_for_training(features)
    if max_observed_at and not output.empty:
        cutoff = pd.to_datetime(max_observed_at)
        observed = pd.to_datetime(output["observed_at"])
        output = output[observed <= cutoff].copy().reset_index(drop=True)
    if not output.empty:
        validate_feature_upload_frame(output)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    summary = {
        "input_csv": input_csv,
        "output_csv": output_csv,
        "raw_rows": int(len(raw)),
        "feature_rows_before_cleaning": int(len(features)),
        "feature_rows_written": int(len(output)),
        "dropped_rows": int(len(features) - len(output)),
        "keep_incomplete_rows": keep_incomplete_rows,
        "max_observed_at": max_observed_at,
        "first_observed_at": None if output.empty else str(output["observed_at"].iloc[0]),
        "last_observed_at": None if output.empty else str(output["observed_at"].iloc[-1]),
        "columns": list(output.columns),
    }
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("built openmeteo features: rows=%s output=%s summary=%s", len(output), output_csv, summary_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="exports/openmeteo_hourly_istanbul_2010_2026_supabase_import_clean.csv")
    parser.add_argument("--output-csv", default="exports/openmeteo_hourly_features_istanbul_2010_2026_supabase_import_clean.csv")
    parser.add_argument("--summary-json", default="exports/openmeteo_hourly_features_summary.json")
    parser.add_argument("--keep-incomplete-rows", action="store_true")
    parser.add_argument("--max-observed-at")
    args = parser.parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        keep_incomplete_rows=args.keep_incomplete_rows,
        max_observed_at=args.max_observed_at,
    )


if __name__ == "__main__":
    main()
