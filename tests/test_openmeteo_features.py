from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weather_ml import openmeteo, training
from weather_ml.openmeteo_features import (
    build_openmeteo_hourly_features,
    clean_features_for_training,
)
from weather_ml.pipelines import download_atmospheric_enrichment_backfill
from weather_ml.pipelines.run_openmeteo_to_supabase import (
    _prepare_feature_context_csv,
    _prepare_raw_upload_csv,
)


def test_fetch_archive_merges_daily_temperatures_into_hourly_rows(monkeypatch) -> None:
    class FakeVariable:
        def __init__(self, values):
            self.values = values

        def ValuesAsNumpy(self):
            return np.array(self.values)

    class FakeHourly:
        def Time(self):
            return 1_767_225_600

        def TimeEnd(self):
            return 1_767_232_800

        def Interval(self):
            return 3600

        def Variables(self, index):
            return FakeVariable([float(index), float(index)])

    class FakeDaily:
        def Time(self):
            return 1_767_225_600

        def TimeEnd(self):
            return 1_767_312_000

        def Interval(self):
            return 86400

        def Variables(self, index):
            return FakeVariable([[9.5], [2.5]][index])

    class FakeResponse:
        def Hourly(self):
            return FakeHourly()

        def Daily(self):
            return FakeDaily()

        def Latitude(self):
            return 41.0

        def Longitude(self):
            return 29.0

        def Elevation(self):
            return 32.0

        def UtcOffsetSeconds(self):
            return 0

    class FakeClient:
        def __init__(self, session):
            pass

        def weather_api(self, url, params):
            assert "cloud_cover_low" in params["hourly"]
            assert params["daily"] == ["temperature_2m_max", "temperature_2m_min"]
            return [FakeResponse()]

    monkeypatch.setattr(openmeteo.requests_cache, "CachedSession", lambda *args, **kwargs: None)
    monkeypatch.setattr(openmeteo, "retry", lambda session, **kwargs: session)
    monkeypatch.setattr(openmeteo.openmeteo_requests, "Client", FakeClient)

    output = openmeteo.fetch_archive_hourly(
        latitude=41.0138,
        longitude=28.9497,
        start_date="2026-01-01",
        end_date="2026-01-01",
    ).hourly

    assert output["temperature_2m_max"].tolist() == [9.5, 9.5]
    assert output["temperature_2m_min"].tolist() == [2.5, 2.5]
    assert output["cloud_cover_low"].tolist() == [9.0, 9.0]


def test_fetch_historical_forecast_preserves_hour_alignment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeVariable:
        def __init__(self, values):
            self.values = values

        def ValuesAsNumpy(self):
            return np.array(self.values)

    class FakeHourly:
        def Time(self):
            return 1_767_225_600

        def TimeEnd(self):
            return 1_767_232_800

        def Interval(self):
            return 3600

        def Variables(self, index):
            return [
                FakeVariable([12.0, np.nan]),
                FakeVariable([1250.0, np.nan]),
                FakeVariable([1.4, np.nan]),
            ][index]

    class FakeClient:
        def __init__(self, session):
            pass

        def weather_api(self, url, params):
            captured["url"] = url
            captured["params"] = params
            return [SimpleNamespace(Hourly=lambda: FakeHourly())]

    monkeypatch.setattr(openmeteo.requests_cache, "CachedSession", lambda *args, **kwargs: None)
    monkeypatch.setattr(openmeteo, "retry", lambda session, **kwargs: session)
    monkeypatch.setattr(openmeteo.openmeteo_requests, "Client", FakeClient)

    output = openmeteo.fetch_historical_forecast_hourly(
        latitude=41.0138,
        longitude=28.9497,
        start_date="2026-01-01",
        end_date="2026-01-01",
    )

    assert captured["url"] == openmeteo.HISTORICAL_FORECAST_URL
    assert captured["params"]["hourly"] == ["cape", "freezing_level_height", "uv_index"]
    assert output["observed_at"].tolist() == ["2026-01-01 00:00:00", "2026-01-01 01:00:00"]
    assert output.iloc[0]["freezing_level_height"] == 1250.0
    assert output.iloc[0]["cape"] == 12.0
    assert output.iloc[0]["uv_index"] == 1.4


def test_openmeteo_features_add_weather_condition_lags_and_24h_target() -> None:
    rows = [
        {
            "id": hour + 1,
            "observed_at": pd.Timestamp("2026-05-01") + pd.Timedelta(hours=hour),
            "temperature_2m": float(hour),
            "relative_humidity_2m": 70.0,
            "dew_point_2m": 10.0,
            "temperature_2m_max": 23.0 if hour < 24 else 27.0,
            "temperature_2m_min": 4.0 if hour < 24 else 8.0,
            "cape": 10.0 if hour == 25 else None,
            "freezing_level_height": 1500.0 if hour == 25 else None,
            "uv_index": 1.2 if hour == 25 else None,
            "apparent_temperature": float(hour),
            "rain": 0.0,
            "weather_code": 0 if hour < 24 else 61,
            "pressure_msl": 1010.0,
            "surface_pressure": 1008.0,
            "cloud_cover": 20.0,
            "cloud_cover_low": 10.0,
            "cloud_cover_mid": 7.0,
            "cloud_cover_high": 3.0,
            "wind_speed_10m": 4.0,
            "wind_direction_10m": 180.0,
            "wind_gusts_10m": 8.0,
        }
        for hour in range(50)
    ]

    features = build_openmeteo_hourly_features(pd.DataFrame(rows))
    cleaned = clean_features_for_training(features)

    assert features.iloc[0]["weather_condition"] == "clear"
    assert features.iloc[0]["target_weather_condition_24h"] == "rain"
    assert features.iloc[28]["weather_condition_lag_4h"] == "rain"
    assert features.iloc[36]["weather_condition_lag_12h"] == "rain"
    assert "target_temperature_24h" not in features.columns
    assert "target_rain_24h" not in features.columns
    assert "target_weather_code_24h" not in features.columns
    assert cleaned["target_weather_condition_24h"].isna().sum() == 0
    assert cleaned["weather_condition_lag_4h"].isna().sum() == 0
    assert cleaned["weather_condition_lag_12h"].isna().sum() == 0
    assert "temp_lag_24h" in cleaned.columns
    assert "freezing_level_height" in cleaned.columns
    assert cleaned["freezing_level_height"].notna().sum() == 1
    assert cleaned.iloc[0]["dew_point_spread"] == 14.0
    assert cleaned.iloc[0]["temperature_2m_max_lag_1d"] == 23.0
    assert cleaned.iloc[0]["temperature_2m_min_lag_1d"] == 4.0
    assert np.isclose(cleaned.iloc[0]["wind_u_10m"], 0.0, atol=1e-8)
    assert np.isclose(cleaned.iloc[0]["wind_v_10m"], 4.0)
    assert "wind_direction_10m" not in training.DEFAULT_FEATURE_COLUMNS
    assert "dew_point_spread" in training.DEFAULT_FEATURE_COLUMNS
    assert "cloud_cover_low" in training.DEFAULT_FEATURE_COLUMNS
    assert "freezing_level_height" not in training.DEFAULT_FEATURE_COLUMNS
    assert "cape" not in training.DEFAULT_FEATURE_COLUMNS
    assert "uv_index" not in training.DEFAULT_FEATURE_COLUMNS


def test_prepare_raw_upload_deduplicates_observed_at(tmp_path) -> None:
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "upload.csv"
    pd.DataFrame(
        [
            {
                "observed_at": "2026-05-01 00:00:00+00:00",
                "temperature_2m": 10.0,
                "weather_code": 0,
            },
            {
                "observed_at": "2026-05-01 00:00:00+00:00",
                "temperature_2m": 11.0,
                "weather_code": 61,
            },
        ]
    ).to_csv(input_csv, index=False)

    _prepare_raw_upload_csv(str(input_csv), str(output_csv))
    output = pd.read_csv(output_csv)

    assert len(output) == 1
    assert output.iloc[0]["temperature_2m"] == 11.0
    assert output.iloc[0]["weather_condition"] == "rain"
    assert "freezing_level_height" not in output.columns


def test_prepare_feature_context_merges_forecast_fields_by_observed_at(tmp_path) -> None:
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "feature_context.csv"
    pd.DataFrame(
        [
            {"observed_at": "2026-05-01 00:00:00+00:00", "weather_code": 0},
            {"observed_at": "2026-05-01 01:00:00+00:00", "weather_code": 61},
        ]
    ).to_csv(input_csv, index=False)
    forecast_context = pd.DataFrame(
        [
            {
                "observed_at": "2026-05-01 01:00:00",
                "cape": 30.0,
                "freezing_level_height": 920.0,
                "uv_index": 2.5,
            }
        ]
    )

    _prepare_feature_context_csv(str(input_csv), str(output_csv), forecast_context)
    output = pd.read_csv(output_csv)

    assert pd.isna(output.iloc[0]["freezing_level_height"])
    assert output.iloc[1]["freezing_level_height"] == 920.0
    assert output.iloc[1]["cape"] == 30.0
    assert output.iloc[1]["uv_index"] == 2.5


def test_combined_backfill_export_left_joins_forecast_values_by_hour(monkeypatch, tmp_path) -> None:
    output_csv = tmp_path / "atmospheric_enrichment.csv"
    monkeypatch.setattr(
        download_atmospheric_enrichment_backfill,
        "get_settings",
        lambda: SimpleNamespace(
            openmeteo_lat=41.0138,
            openmeteo_lon=28.9497,
            openmeteo_start_date="2010-01-01",
            openmeteo_end_date="2026-05-25",
        ),
    )
    monkeypatch.setattr(
        download_atmospheric_enrichment_backfill,
        "fetch_archive_hourly",
        lambda **kwargs: SimpleNamespace(
            hourly=pd.DataFrame(
                [
                    {
                        "observed_at": "2021-03-22 23:00:00+00:00",
                        "cloud_cover_low": 10.0,
                        "cloud_cover_mid": 20.0,
                        "cloud_cover_high": 30.0,
                        "temperature_2m_max": 11.0,
                        "temperature_2m_min": 5.0,
                    },
                    {
                        "observed_at": "2021-03-23 00:00:00+00:00",
                        "cloud_cover_low": 12.0,
                        "cloud_cover_mid": 22.0,
                        "cloud_cover_high": 32.0,
                        "temperature_2m_max": 12.0,
                        "temperature_2m_min": 6.0,
                    },
                ]
            )
        ),
    )
    monkeypatch.setattr(
        download_atmospheric_enrichment_backfill,
        "fetch_historical_forecast_hourly",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "observed_at": "2021-03-23 00:00:00",
                    "cape": 10.0,
                    "freezing_level_height": 740.0,
                    "uv_index": 1.5,
                },
            ]
        ),
    )

    download_atmospheric_enrichment_backfill.run(output_csv=str(output_csv))
    output = pd.read_csv(output_csv)

    assert Path(output_csv).exists()
    assert len(output) == 2
    assert pd.isna(output.iloc[0]["cape"])
    assert output.iloc[1]["cloud_cover_low"] == 12.0
    assert output.iloc[1]["cape"] == 10.0
    assert output.iloc[1]["freezing_level_height"] == 740.0
    assert output.iloc[1]["uv_index"] == 1.5


def test_combined_enrichment_sql_adds_columns_and_updates_both_tables() -> None:
    sql = Path("sql/006_apply_atmospheric_enrichment_backfill.sql").read_text(encoding="utf-8")

    assert '"KadikoyWeatherCodeRaw"' in sql
    assert "add column if not exists cloud_cover_low" in sql
    assert "add column if not exists temperature_2m_max" in sql
    assert '"KadikoyWeatherCodeFeature"' in sql
    assert "add column if not exists dew_point_spread" in sql
    assert "add column if not exists wind_u_10m" in sql
    assert "add column if not exists cape" in sql
    assert "add column if not exists uv_index" in sql
    assert "kadikoy_atmospheric_enrichment_backfill.csv" in sql
    assert '"KadikoyAtmosphericEnrichmentStaging"' in sql
    assert "\\copy" not in sql
    assert 'update "KadikoyWeatherCodeRaw"' in sql
    assert "cloud_cover_low = source.cloud_cover_low" in sql
    assert "temperature_2m_min = source.temperature_2m_min" in sql
    assert 'update "KadikoyWeatherCodeFeature"' in sql
    assert "freezing_level_height = source.freezing_level_height" in sql
    assert "cape = source.cape" in sql
    assert "uv_index = source.uv_index" in sql
    assert "dew_point_spread = feature.temperature_2m - feature.dew_point_2m" in sql
    assert "wind_u_10m = -" in sql
    assert "temperature_2m_max_lag_1d" in sql
    assert "interval '1 day'" in sql
    assert "target_weather_condition_24h" not in sql
