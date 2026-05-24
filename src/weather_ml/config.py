from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    supabase_db_url: str | None = Field(None, alias="SUPABASE_DB_URL")
    supabase_url: str | None = Field(None, alias="SUPABASE_URL")
    supabase_key: str | None = Field(None, alias="SUPABASE_KEY")
    supabase_service_role_key: str | None = Field(None, alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_project_ref: str = Field("etjsdhmpgxtyfqezoejz", alias="SUPABASE_PROJECT_REF")
    supabase_access_token: str | None = Field(None, alias="SUPABASE_ACCESS_TOKEN")

    weather_location_name: str = Field("Kadikoy", alias="WEATHER_LOCATION_NAME")
    weather_timezone: str = Field("Europe/Istanbul", alias="WEATHER_TIMEZONE")
    weather_forecast_hours: int = Field(48, alias="WEATHER_FORECAST_HOURS")

    openmeteo_lat: float = Field(41.0138, alias="OPENMETEO_LAT")
    openmeteo_lon: float = Field(28.9497, alias="OPENMETEO_LON")
    openmeteo_location_name: str = Field("Istanbul", alias="OPENMETEO_LOCATION_NAME")
    openmeteo_start_date: str = Field("2010-01-01", alias="OPENMETEO_START_DATE")
    openmeteo_end_date: str = Field("2026-05-22", alias="OPENMETEO_END_DATE")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def require_supabase_db_url() -> str:
    value = get_settings().supabase_db_url
    if not value:
        raise RuntimeError("SUPABASE_DB_URL is required for Supabase write operations.")
    return value


def require_supabase_rest_credentials() -> tuple[str, str]:
    settings = get_settings()
    key = settings.supabase_service_role_key or settings.supabase_key
    if not settings.supabase_url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY are required for Supabase REST operations.")
    return settings.supabase_url, key


def require_supabase_management_credentials() -> tuple[str, str]:
    settings = get_settings()
    if not settings.supabase_access_token:
        raise RuntimeError("SUPABASE_ACCESS_TOKEN is required for Supabase Management API SQL operations.")
    return settings.supabase_project_ref, settings.supabase_access_token
