from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from weather_ml.config import get_settings
from weather_ml.openmeteo import fetch_archive_hourly, fetch_historical_forecast_hourly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATMOSPHERIC_ENRICHMENT_COLUMNS = [
    "observed_at",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m_max",
    "temperature_2m_min",
    "cape",
    "freezing_level_height",
    "uv_index",
]


def run(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_csv: str = "exports/kadikoy_atmospheric_enrichment_backfill.csv",
) -> None:
    settings = get_settings()
    selected_start = start_date or settings.openmeteo_start_date
    selected_end = end_date or settings.openmeteo_end_date
    archive = (
        fetch_archive_hourly(
            latitude=settings.openmeteo_lat,
            longitude=settings.openmeteo_lon,
            start_date=selected_start,
            end_date=selected_end,
        )
        .hourly[
            [
                "observed_at",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                "temperature_2m_max",
                "temperature_2m_min",
            ]
        ]
        .copy()
    )
    forecast = fetch_historical_forecast_hourly(
        latitude=settings.openmeteo_lat,
        longitude=settings.openmeteo_lon,
        start_date=selected_start,
        end_date=selected_end,
    ).copy()

    archive["observed_at"] = pd.to_datetime(archive["observed_at"], utc=True).dt.tz_localize(None)
    forecast["observed_at"] = pd.to_datetime(forecast["observed_at"], utc=True).dt.tz_localize(None)
    output = archive.merge(forecast, on="observed_at", how="left", validate="one_to_one")
    output["observed_at"] = output["observed_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output = output[ATMOSPHERIC_ENRICHMENT_COLUMNS]

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    logger.info(
        "downloaded atmospheric enrichment backfill: rows=%s forecast_rows=%s output=%s",
        len(output),
        int(output[["cape", "freezing_level_height", "uv_index"]].notna().any(axis=1).sum()),
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--output-csv",
        default="exports/kadikoy_atmospheric_enrichment_backfill.csv",
    )
    args = parser.parse_args()
    run(start_date=args.start_date, end_date=args.end_date, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
