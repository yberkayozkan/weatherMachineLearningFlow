from __future__ import annotations

from dataclasses import dataclass

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OPENMETEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "rain",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


@dataclass(frozen=True)
class OpenMeteoArchiveResult:
    metadata: dict[str, float | int]
    hourly: pd.DataFrame


def fetch_archive_hourly(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_path: str = ".cache/openmeteo",
) -> OpenMeteoArchiveResult:
    cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    responses = openmeteo.weather_api(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": OPENMETEO_HOURLY_VARIABLES,
        },
    )
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "observed_at": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for index, variable_name in enumerate(OPENMETEO_HOURLY_VARIABLES):
        hourly_data[variable_name] = hourly.Variables(index).ValuesAsNumpy()

    frame = pd.DataFrame(data=hourly_data)
    metadata = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation": response.Elevation(),
        "utc_offset_seconds": response.UtcOffsetSeconds(),
    }
    return OpenMeteoArchiveResult(metadata=metadata, hourly=frame)


def normalize_archive_hourly(
    result: OpenMeteoArchiveResult,
    *,
    location_name: str,
) -> pd.DataFrame:
    frame = result.hourly.copy()
    frame.insert(0, "source", "openmeteo_archive")
    frame.insert(1, "location_name", location_name)
    frame.insert(2, "latitude", result.metadata["latitude"])
    frame.insert(3, "longitude", result.metadata["longitude"])
    frame.insert(4, "elevation", result.metadata["elevation"])
    frame.insert(5, "utc_offset_seconds", result.metadata["utc_offset_seconds"])
    return frame

