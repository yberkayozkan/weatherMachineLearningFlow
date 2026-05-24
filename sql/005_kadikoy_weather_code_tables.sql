create table if not exists "KadikoyWeatherCodeRaw" (
    id bigint primary key,
    latitude double precision,
    longitude double precision,
    elevation double precision,
    utc_offset_seconds integer,
    observed_at timestamp not null unique,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    dew_point_2m double precision,
    apparent_temperature double precision,
    rain double precision,
    weather_code integer,
    pressure_msl double precision,
    surface_pressure double precision,
    cloud_cover double precision,
    wind_speed_10m double precision,
    wind_direction_10m double precision,
    wind_gusts_10m double precision,
    weather_condition text
);

create index if not exists idx_kadikoy_weather_code_raw_observed
    on "KadikoyWeatherCodeRaw" (observed_at desc);

create table if not exists "KadikoyWeatherCodeFeature" (
    id bigint primary key,
    observed_at timestamp not null unique,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    dew_point_2m double precision,
    apparent_temperature double precision,
    rain double precision,
    pressure_msl double precision,
    surface_pressure double precision,
    cloud_cover double precision,
    wind_speed_10m double precision,
    wind_direction_10m double precision,
    wind_gusts_10m double precision,
    weather_code integer,
    weather_condition text,
    weather_code_lag_4h integer,
    weather_code_lag_12h integer,
    weather_condition_lag_4h text,
    weather_condition_lag_12h text,
    weather_condition_lag_4h_code integer,
    weather_condition_lag_12h_code integer,
    temp_lag_1h double precision,
    temp_lag_3h double precision,
    temp_lag_6h double precision,
    temp_lag_24h double precision,
    temp_rolling_mean_3h double precision,
    temp_rolling_mean_6h double precision,
    temp_rolling_mean_24h double precision,
    temp_rolling_std_24h double precision,
    humidity_lag_1h double precision,
    humidity_rolling_mean_6h double precision,
    pressure_lag_1h double precision,
    pressure_change_3h double precision,
    wind_speed_lag_1h double precision,
    wind_speed_rolling_mean_6h double precision,
    rain_rolling_sum_6h double precision,
    rain_rolling_sum_24h double precision,
    hour_of_day integer,
    day_of_week integer,
    month integer,
    day_of_year integer,
    is_weekend boolean,
    target_weather_condition_24h text not null
);

create index if not exists idx_kadikoy_weather_code_feature_observed
    on "KadikoyWeatherCodeFeature" (observed_at desc);

create index if not exists idx_kadikoy_weather_code_feature_condition
    on "KadikoyWeatherCodeFeature" (weather_condition);

create index if not exists idx_kadikoy_weather_code_feature_target_condition
    on "KadikoyWeatherCodeFeature" (target_weather_condition_24h);

create index if not exists idx_kadikoy_weather_code_feature_lag_conditions
    on "KadikoyWeatherCodeFeature" (weather_condition_lag_4h, weather_condition_lag_12h);
