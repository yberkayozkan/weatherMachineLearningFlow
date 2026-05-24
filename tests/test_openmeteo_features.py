import pandas as pd

from weather_ml.openmeteo_features import build_openmeteo_hourly_features, clean_features_for_training
from weather_ml.pipelines.run_openmeteo_to_supabase import _prepare_raw_upload_csv


def test_openmeteo_features_add_weather_condition_lags_and_24h_target() -> None:
    rows = [
        {
            "id": hour + 1,
            "observed_at": pd.Timestamp("2026-05-01") + pd.Timedelta(hours=hour),
            "temperature_2m": float(hour),
            "relative_humidity_2m": 70.0,
            "dew_point_2m": 10.0,
            "apparent_temperature": float(hour),
            "rain": 0.0,
            "weather_code": 0 if hour < 24 else 61,
            "pressure_msl": 1010.0,
            "surface_pressure": 1008.0,
            "cloud_cover": 20.0,
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
