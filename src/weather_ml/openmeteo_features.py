from __future__ import annotations

import pandas as pd


REQUIRED_OPENMETEO_COLUMNS = [
    "observed_at",
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

REQUIRED_MODEL_COLUMNS = [
    "target_weather_condition_24h",
    "weather_condition_lag_4h_code",
    "weather_condition_lag_12h_code",
    "temp_lag_24h",
    "temp_rolling_std_24h",
]

FEATURE_UPLOAD_COLUMNS = [
    "id",
    "observed_at",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "rain",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "weather_code",
    "weather_condition",
    "weather_code_lag_4h",
    "weather_code_lag_12h",
    "weather_condition_lag_4h",
    "weather_condition_lag_12h",
    "weather_condition_lag_4h_code",
    "weather_condition_lag_12h_code",
    "temp_lag_1h",
    "temp_lag_3h",
    "temp_lag_6h",
    "temp_lag_24h",
    "temp_rolling_mean_3h",
    "temp_rolling_mean_6h",
    "temp_rolling_mean_24h",
    "temp_rolling_std_24h",
    "humidity_lag_1h",
    "humidity_rolling_mean_6h",
    "pressure_lag_1h",
    "pressure_change_3h",
    "wind_speed_lag_1h",
    "wind_speed_rolling_mean_6h",
    "rain_rolling_sum_6h",
    "rain_rolling_sum_24h",
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "target_weather_condition_24h",
]

FORBIDDEN_FUTURE_TARGET_COLUMNS = [
    "target_temperature_24h",
    "target_rain_24h",
    "target_weather_code_24h",
]

WEATHER_CONDITION_LABELS = [
    "clear",
    "cloudy",
    "fog",
    "drizzle",
    "rain",
    "freezing_rain",
    "snow",
    "showers",
    "thunderstorm",
]

WEATHER_CODE_TO_CONDITION = {
    0: "clear",
    1: "cloudy",
    2: "cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    56: "drizzle",
    57: "drizzle",
    61: "rain",
    63: "rain",
    65: "rain",
    66: "freezing_rain",
    67: "freezing_rain",
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",
    80: "showers",
    81: "showers",
    82: "showers",
    85: "showers",
    86: "showers",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def map_weather_code_to_condition(code: object) -> str | None:
    if pd.isna(code):
        return None
    try:
        normalized = int(round(float(code)))
    except (TypeError, ValueError):
        return None
    return WEATHER_CODE_TO_CONDITION.get(normalized)


def build_openmeteo_hourly_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    missing = [column for column in REQUIRED_OPENMETEO_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required Open-Meteo columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    frame = frame.sort_values("observed_at").reset_index(drop=True)

    for column in [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "rain",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    features = pd.DataFrame()
    if "id" in frame.columns:
        features["id"] = pd.to_numeric(frame["id"], errors="coerce").astype("Int64")
    else:
        features["id"] = range(1, len(frame) + 1)
    features["observed_at"] = frame["observed_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    features["temperature_2m"] = frame["temperature_2m"]
    features["relative_humidity_2m"] = frame["relative_humidity_2m"]
    features["dew_point_2m"] = frame["dew_point_2m"]
    features["apparent_temperature"] = frame["apparent_temperature"]
    features["rain"] = frame["rain"].fillna(0)
    features["pressure_msl"] = frame["pressure_msl"]
    features["surface_pressure"] = frame["surface_pressure"]
    features["cloud_cover"] = frame["cloud_cover"]
    features["wind_speed_10m"] = frame["wind_speed_10m"]
    features["wind_direction_10m"] = frame["wind_direction_10m"]
    features["wind_gusts_10m"] = frame["wind_gusts_10m"]
    features["weather_code"] = pd.to_numeric(frame["weather_code"], errors="coerce").round().astype("Int64")
    features["weather_condition"] = features["weather_code"].map(map_weather_code_to_condition)
    features["weather_code_lag_4h"] = features["weather_code"].shift(4)
    features["weather_code_lag_12h"] = features["weather_code"].shift(12)
    features["weather_condition_lag_4h"] = features["weather_code_lag_4h"].map(map_weather_code_to_condition)
    features["weather_condition_lag_12h"] = features["weather_code_lag_12h"].map(map_weather_code_to_condition)
    condition_to_code = {condition: index for index, condition in enumerate(WEATHER_CONDITION_LABELS)}
    features["weather_condition_lag_4h_code"] = features["weather_condition_lag_4h"].map(condition_to_code).astype("Int64")
    features["weather_condition_lag_12h_code"] = features["weather_condition_lag_12h"].map(condition_to_code).astype("Int64")

    features["temp_lag_1h"] = frame["temperature_2m"].shift(1)
    features["temp_lag_3h"] = frame["temperature_2m"].shift(3)
    features["temp_lag_6h"] = frame["temperature_2m"].shift(6)
    features["temp_lag_24h"] = frame["temperature_2m"].shift(24)
    features["temp_rolling_mean_3h"] = frame["temperature_2m"].rolling(3, min_periods=1).mean()
    features["temp_rolling_mean_6h"] = frame["temperature_2m"].rolling(6, min_periods=1).mean()
    features["temp_rolling_mean_24h"] = frame["temperature_2m"].rolling(24, min_periods=1).mean()
    features["temp_rolling_std_24h"] = frame["temperature_2m"].rolling(24, min_periods=2).std()

    features["humidity_lag_1h"] = frame["relative_humidity_2m"].shift(1)
    features["humidity_rolling_mean_6h"] = frame["relative_humidity_2m"].rolling(6, min_periods=1).mean()
    features["pressure_lag_1h"] = frame["pressure_msl"].shift(1)
    features["pressure_change_3h"] = frame["pressure_msl"] - frame["pressure_msl"].shift(3)
    features["wind_speed_lag_1h"] = frame["wind_speed_10m"].shift(1)
    features["wind_speed_rolling_mean_6h"] = frame["wind_speed_10m"].rolling(6, min_periods=1).mean()
    features["rain_rolling_sum_6h"] = features["rain"].rolling(6, min_periods=1).sum()
    features["rain_rolling_sum_24h"] = features["rain"].rolling(24, min_periods=1).sum()

    observed = frame["observed_at"].dt
    features["hour_of_day"] = observed.hour
    features["day_of_week"] = observed.dayofweek
    features["month"] = observed.month
    features["day_of_year"] = observed.dayofyear
    features["is_weekend"] = features["day_of_week"].isin([5, 6])
    features["target_weather_condition_24h"] = features["weather_condition"].shift(-24)

    return features


def clean_features_for_training(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features

    missing = [column for column in REQUIRED_MODEL_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {', '.join(missing)}")

    output = features.dropna(subset=REQUIRED_MODEL_COLUMNS).copy().reset_index(drop=True)
    return output[FEATURE_UPLOAD_COLUMNS]


def validate_feature_upload_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_UPLOAD_COLUMNS if column not in frame.columns]
    extra_future_targets = [column for column in FORBIDDEN_FUTURE_TARGET_COLUMNS if column in frame.columns]
    if missing or extra_future_targets:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra_future_targets:
            details.append(f"forbidden future target columns: {', '.join(extra_future_targets)}")
        raise ValueError("Invalid weather feature frame; " + "; ".join(details))

    if list(frame.columns) != FEATURE_UPLOAD_COLUMNS:
        raise ValueError("Invalid weather feature frame; CSV columns must match the Supabase feature schema order.")
