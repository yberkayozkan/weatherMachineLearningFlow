from __future__ import annotations

import argparse
import logging
from pathlib import Path

from weather_ml.config import get_settings
from weather_ml.local_store import timestamp_slug, write_dataframe, write_json
from weather_ml.openmeteo import fetch_archive_hourly, normalize_archive_hourly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(
    *,
    output_dir: str = "data/openmeteo_archive",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    settings = get_settings()
    start = start_date or settings.openmeteo_start_date
    end = end_date or settings.openmeteo_end_date
    slug = timestamp_slug()
    base_dir = Path(output_dir) / settings.openmeteo_location_name / slug

    result = fetch_archive_hourly(
        latitude=settings.openmeteo_lat,
        longitude=settings.openmeteo_lon,
        start_date=start,
        end_date=end,
    )
    hourly = normalize_archive_hourly(result, location_name=settings.openmeteo_location_name)

    metadata = {
        **result.metadata,
        "location_name": settings.openmeteo_location_name,
        "start_date": start,
        "end_date": end,
        "rows": len(hourly),
    }
    write_json(base_dir / "metadata.json", metadata)
    write_dataframe(base_dir / "openmeteo_hourly.csv", hourly)

    latest_dir = Path(output_dir) / settings.openmeteo_location_name / "latest"
    write_json(latest_dir / "metadata.json", metadata)
    write_dataframe(latest_dir / "openmeteo_hourly.csv", hourly)

    logger.info("downloaded openmeteo archive: rows=%s output=%s", len(hourly), base_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-dir", default="data/openmeteo_archive")
    args = parser.parse_args()
    run(output_dir=args.output_dir, start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()

