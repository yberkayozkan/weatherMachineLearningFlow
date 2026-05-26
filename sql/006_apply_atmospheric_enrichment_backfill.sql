-- Supabase SQL Editor compatible atmospheric enrichment update.
-- Run this script once to create/alter tables, import
-- exports/kadikoy_atmospheric_enrichment_backfill.csv into
-- "KadikoyAtmosphericEnrichmentStaging" through Table Editor, then run this
-- same script again to apply the loaded staging values.

begin;

alter table "KadikoyWeatherCodeRaw"
    add column if not exists cloud_cover_low double precision,
    add column if not exists cloud_cover_mid double precision,
    add column if not exists cloud_cover_high double precision,
    add column if not exists temperature_2m_max double precision,
    add column if not exists temperature_2m_min double precision;

alter table "KadikoyWeatherCodeFeature"
    add column if not exists cloud_cover_low double precision,
    add column if not exists cloud_cover_mid double precision,
    add column if not exists cloud_cover_high double precision,
    add column if not exists temperature_2m_max double precision,
    add column if not exists temperature_2m_min double precision,
    add column if not exists temperature_2m_max_lag_1d double precision,
    add column if not exists temperature_2m_min_lag_1d double precision,
    add column if not exists dew_point_spread double precision,
    add column if not exists wind_u_10m double precision,
    add column if not exists wind_v_10m double precision,
    add column if not exists cape double precision,
    add column if not exists freezing_level_height double precision,
    add column if not exists uv_index double precision;

create table if not exists "KadikoyAtmosphericEnrichmentStaging" (
    observed_at timestamp primary key,
    cloud_cover_low double precision,
    cloud_cover_mid double precision,
    cloud_cover_high double precision,
    temperature_2m_max double precision,
    temperature_2m_min double precision,
    cape double precision,
    freezing_level_height double precision,
    uv_index double precision
);

alter table "KadikoyAtmosphericEnrichmentStaging" enable row level security;

update "KadikoyWeatherCodeRaw" as raw
set cloud_cover_low = source.cloud_cover_low,
    cloud_cover_mid = source.cloud_cover_mid,
    cloud_cover_high = source.cloud_cover_high,
    temperature_2m_max = source.temperature_2m_max,
    temperature_2m_min = source.temperature_2m_min
from "KadikoyAtmosphericEnrichmentStaging" as source
where raw.observed_at = source.observed_at;

update "KadikoyWeatherCodeFeature" as feature
set cloud_cover_low = source.cloud_cover_low,
    cloud_cover_mid = source.cloud_cover_mid,
    cloud_cover_high = source.cloud_cover_high,
    temperature_2m_max = source.temperature_2m_max,
    temperature_2m_min = source.temperature_2m_min,
    cape = source.cape,
    freezing_level_height = source.freezing_level_height,
    uv_index = source.uv_index,
    dew_point_spread = feature.temperature_2m - feature.dew_point_2m,
    wind_u_10m = -feature.wind_speed_10m * sin(radians(feature.wind_direction_10m)),
    wind_v_10m = -feature.wind_speed_10m * cos(radians(feature.wind_direction_10m))
from "KadikoyAtmosphericEnrichmentStaging" as source
where feature.observed_at = source.observed_at;

update "KadikoyWeatherCodeFeature" as feature
set temperature_2m_max_lag_1d = previous_day.temperature_2m_max,
    temperature_2m_min_lag_1d = previous_day.temperature_2m_min
from "KadikoyWeatherCodeFeature" as previous_day
where previous_day.observed_at = feature.observed_at - interval '1 day';

commit;
