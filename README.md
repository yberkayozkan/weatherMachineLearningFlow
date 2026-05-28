# Weather Machine Learning

An hourly weather data pipeline that retrieves Open-Meteo archive observations,
engineers time-based features, persists raw and feature-ready records in
Supabase, and trains XGBoost or LightGBM classifiers to predict the weather
condition 24 hours ahead.

## Pipeline

1. Retrieve hourly and daily Historical Weather observations from Open-Meteo,
   plus nullable `cape`, `freezing_level_height`, and `uv_index` enrichment
   from the Historical Forecast API for feature rows.
2. Map WMO weather codes into condition groups and build lag, rolling-window,
   calendar, dew-point-spread, wind-vector, and prior-day temperature features.
3. Upsert raw observations and target-safe feature rows into Supabase tables
   keyed by `observed_at`.
4. Train an `XGBClassifier` or `LGBMClassifier` for the multiclass
   `target_weather_condition_24h` target using 3-fold expanding-window
   development validation and a latest-period final holdout. A separate
   XGBoost profile uses Historical Forecast enrichment from 2021-03-23 onward.
5. Manually log model parameters, summary metrics, run tags, iteration-level
   train/test metrics, and generated artifacts to MLflow.

The only future label stored in the feature table is
`target_weather_condition_24h`; numeric future targets and future weather-code
targets are excluded. Training applies a fixed 24-hour gap before every
development validation fold and before the final holdout, matching the
24-hour forecast horizon and preventing boundary leakage.
`cape`, `freezing_level_height`, and `uv_index` are excluded from full-history
XGBoost and LightGBM training because their Historical Forecast coverage does
not span the full history. They are included in the separate Historical
Forecast XGBoost profile, which restricts training data to timestamps on or
after `2021-03-23 00:00:00`. Daily maximum/minimum temperatures are stored as
source values, while only their previous-day lag features are used for
training to avoid same-day lookahead.

```text
clear
cloudy
fog
drizzle
rain
freezing_rain
snow
showers
thunderstorm
```

## Technologies

- Python, pandas, and NumPy for data processing and feature generation.
- Open-Meteo API client with request caching and retry support for archive retrieval.
- Supabase with PostgreSQL/SQLAlchemy access paths for storage and upserts.
- XGBoost, LightGBM, and scikit-learn for multiclass model training and evaluation.
- Matplotlib for evaluation and feature-importance visual artifacts.
- MLflow configured for DagsHub-compatible tracking of parameters, metrics,
  tags, metric history, and artifacts.
- GitHub Actions for ingestion and training workflows.
- pytest and Ruff for verification and code-quality checks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `.env` with Supabase and MLflow credentials. Do not commit `.env`.

Required Supabase values:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

Required DagsHub MLflow values:

```text
MLFLOW_TRACKING_URI=https://dagshub.com/<dagshub_username>/<dagshub_repo>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub_username>
MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

## Supabase Tables

Create the canonical tables with:

```powershell
psql "$env:SUPABASE_DB_URL" -f sql/005_kadikoy_weather_code_tables.sql
```

For an existing database, generate the single atmospheric enrichment CSV:

```powershell
python -m weather_ml.pipelines.download_atmospheric_enrichment_backfill --start-date 2010-01-01 --end-date 2026-05-25
```

For Supabase Dashboard SQL Editor:

1. Run `sql/006_apply_atmospheric_enrichment_backfill.sql` once to add the
   columns and create `KadikoyAtmosphericEnrichmentStaging`.
2. In Table Editor, import
   `exports/kadikoy_atmospheric_enrichment_backfill.csv` into
   `KadikoyAtmosphericEnrichmentStaging`.
3. Run the same SQL file again to update the raw and feature tables.

The SQL Editor cannot execute `psql` client commands such as `\copy`. Direct
one-command local CSV loading requires a PostgreSQL connection and `psql`.

Default table names:

```text
KadikoyWeatherCodeRaw
KadikoyWeatherCodeFeature
```

### Data Dictionary

`KadikoyWeatherCodeRaw` stores hourly observations prepared from the Open-Meteo
archive feed.

| Column(s) | Type | Description |
| --- | --- | --- |
| `id` | `bigint` | Hour-based primary key generated for each observation. |
| `observed_at` | `timestamp` | Unique hourly observation time used for upserts. |
| `latitude`, `longitude`, `elevation`, `utc_offset_seconds` | numeric | Location and source-time metadata when included in the raw input. |
| `temperature_2m`, `dew_point_2m`, `apparent_temperature` | `double precision` | Temperature measurements in degrees Celsius. |
| `temperature_2m_max`, `temperature_2m_min` | `double precision` | Daily maximum/minimum temperatures repeated on that calendar day's hourly rows. |
| `relative_humidity_2m`, `cloud_cover` | `double precision` | Hourly humidity and total cloud-cover measurements. |
| `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | `double precision` | Hourly cloud coverage by atmospheric layer. |
| `rain` | `double precision` | Hourly rain measurement. |
| `pressure_msl`, `surface_pressure` | `double precision` | Atmospheric pressure measurements. |
| `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m` | `double precision` | Wind measurements at 10 metres. |
| `weather_code` | `integer` | Open-Meteo/WMO weather code. |
| `weather_condition` | `text` | Weather-code group such as `clear`, `rain`, or `snow`. |

`KadikoyWeatherCodeFeature` stores model-ready observations and engineered
features. Rows without required historical inputs or the 24-hour target are
removed before upload.

| Column(s) | Type | Description |
| --- | --- | --- |
| `id`, `observed_at` | `bigint`, `timestamp` | Primary key and unique hourly timestamp used for upserts. |
| `temperature_2m` through `wind_gusts_10m` | `double precision` | Current-hour weather measurements retained from raw data, including layered cloud cover and daily source extrema. |
| `dew_point_spread` | `double precision` | Current temperature minus dew point: `temperature_2m - dew_point_2m`. |
| `wind_u_10m`, `wind_v_10m` | `double precision` | Wind components computed from speed and direction using sine/cosine transforms. |
| `temperature_2m_max_lag_1d`, `temperature_2m_min_lag_1d` | `double precision` | Previous completed calendar day's extrema, used as leakage-safe model features. |
| `cape` | `double precision` | Convective available potential energy from Historical Forecast, nullable when unavailable. |
| `freezing_level_height` | `double precision` | Height in metres of the 0 degrees Celsius freezing level from Historical Forecast, nullable when unavailable. |
| `uv_index` | `double precision` | UV index from Historical Forecast, nullable when unavailable. |
| `weather_code`, `weather_condition` | `integer`, `text` | Current-hour weather code and mapped condition class. |
| `weather_code_lag_4h`, `weather_code_lag_12h` | `integer` | Weather codes observed 4 and 12 hours earlier. |
| `weather_condition_lag_4h`, `weather_condition_lag_12h` | `text` | Condition groups observed 4 and 12 hours earlier. |
| `weather_condition_lag_4h_code`, `weather_condition_lag_12h_code` | `integer` | Numeric encodings of lagged condition groups used for model training. |
| `temp_lag_1h`, `temp_lag_3h`, `temp_lag_6h`, `temp_lag_24h` | `double precision` | Past temperature values at fixed hourly offsets. |
| `temp_rolling_mean_3h`, `temp_rolling_mean_6h`, `temp_rolling_mean_24h`, `temp_rolling_std_24h` | `double precision` | Rolling temperature statistics. |
| `humidity_lag_1h`, `humidity_rolling_mean_6h` | `double precision` | Humidity lag and rolling average features. |
| `pressure_lag_1h`, `pressure_change_3h` | `double precision` | Lagged pressure and three-hour pressure change. |
| `wind_speed_lag_1h`, `wind_speed_rolling_mean_6h` | `double precision` | Lagged and rolling wind-speed features. |
| `rain_rolling_sum_6h`, `rain_rolling_sum_24h` | `double precision` | Rolling precipitation totals. |
| `hour_of_day`, `day_of_week`, `month`, `day_of_year`, `is_weekend` | integer / boolean | Calendar features derived from `observed_at`. |
| `target_weather_condition_24h` | `text` | Training target: the weather-condition group 24 hours after the observation. |

## Local Commands

Download Open-Meteo archive:

```powershell
python -m weather_ml.pipelines.download_openmeteo_archive
```

Run the full daily-safe pipeline:

```powershell
python -m weather_ml.pipelines.run_openmeteo_to_supabase
```

Loads are idempotent: raw and feature rows are upserted on `observed_at`, so
rerunning the same date window updates existing rows instead of inserting
duplicates. The raw table contains Historical Weather archive fields,
including layered cloud coverage and daily temperature extrema. Historical
Forecast enrichment fields (`cape`, `freezing_level_height`, and `uv_index`)
are joined by `observed_at` only into the feature-table upload.

Generate the single enrichment CSV:

```powershell
python -m weather_ml.pipelines.download_atmospheric_enrichment_backfill --start-date 2010-01-01 --end-date 2026-05-25
```

The CAPE, freezing-level, and UV series come from the Open-Meteo
[Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api).
The Historical Weather `/v1/archive` source used for the base observations
supplies layered clouds and daily temperature extrema, but is not used for
CAPE, freezing-level, or UV enrichment. Hours outside available Historical
Forecast coverage remain `NULL`; no values are imputed. These nullable fields
are model inputs only in the 2021-03-23-and-later Historical Forecast XGBoost
profile. The generated `exports/kadikoy_atmospheric_enrichment_backfill.csv`
contains Archive values for the full selected range and nullable Historical
Forecast values on matching hours; the SQL uses that one staging file to
update both Supabase tables without overwriting targets.

Train from Supabase and log to MLflow:

```powershell
python -m weather_ml.pipelines.train_from_supabase --table-name KadikoyWeatherCodeFeature --cv-splits 3
```

Train from a local feature CSV without MLflow:

```powershell
python -m weather_ml.pipelines.train_from_csv --csv-path path\to\features.csv --cv-splits 3 --no-mlflow
```

Train LightGBM from Supabase and log to its MLflow experiment:

```powershell
python -m weather_ml.pipelines.train_lightgbm_from_supabase --table-name KadikoyWeatherCodeFeature --cv-splits 3
```

Train LightGBM from a local feature CSV without MLflow:

```powershell
python -m weather_ml.pipelines.train_lightgbm_from_csv --csv-path path\to\features.csv --cv-splits 3 --no-mlflow
```

Train the enriched XGBoost profile from Supabase and log it to its separate
MLflow experiment:

```powershell
python -m weather_ml.pipelines.train_historical_forecast_xgboost_from_supabase --table-name KadikoyWeatherCodeFeature --cv-splits 3
```

## Training Outputs

XGBoost writes to `artifacts/training` by default with model file
`weather_condition_xgboost_model.pkl`. LightGBM writes to
`artifacts/lightgbm-training` by default with model file
`weather_condition_lightgbm_model.pkl`. The Historical Forecast XGBoost
profile writes to `artifacts/xgboost-historical-forecast-training` with model
file `weather_condition_xgboost_historical_forecast_model.pkl`. Each training
run also writes the following shared outputs; when MLflow logging is enabled,
the output directory is uploaded to the run.

```text
metrics.json
time_series_cv_metrics.csv
time_series_cv_metrics.png
metrics_history.csv
test_predictions.csv
confusion_matrix.png
feature_importance.csv
feature_importance.png
metrics_history.png
roc_auc_ovr.png (when ROC curves can be calculated)
```

`feature_importance.csv` and `feature_importance.png` report the trained
model's `feature_importances_` values. `time_series_cv_metrics.csv` and its
plot report development validation stability across expanding-window folds;
the acceptance tag is determined only by final holdout `f1_macro`.
Both trainers compute square-root balanced sample weights from each training partition
only, for every cross-validation fold and the final model fit. Active
atmospheric model inputs include dew-point spread, wind U/V components,
layered cloud cover, and previous-day maximum/minimum temperature lags;
same-day extrema are excluded in every profile. The full-history models
exclude nullable Historical Forecast enrichment; the separate XGBoost
Historical Forecast profile adds `cape`, `freezing_level_height`, and
`uv_index`. XGBoost logs to `weather-condition-xgboost`; LightGBM logs to
`weather-condition-lightgbm`; Historical Forecast XGBoost logs to
`weather-condition-xgboost-historical-forecast`.
MLflow metrics are logged explicitly by the training code; MLflow autologging
is not enabled.

## GitHub Actions

`.github/workflows/weather-ingestion.yml` runs on a daily schedule and can be
started manually. It downloads a recent Open-Meteo window, builds features,
and writes raw and feature rows to Supabase.

`.github/workflows/train-model.yml` can be started manually with XGBoost,
final holdout, and `cv_splits` parameters, or runs after a successful ingestion
workflow. It trains
from the Supabase feature table, logs to MLflow, and uploads the training
output directory as a GitHub Actions artifact.

`.github/workflows/train-lightgbm-model.yml` can be started manually with
LightGBM, final holdout, and `cv_splits` parameters, or runs after a successful
ingestion workflow. It trains from the Supabase feature table, logs to the
LightGBM MLflow experiment, and uploads `artifacts/lightgbm-training`.

`.github/workflows/train-xgboost-historical-forecast-model.yml` can be
started manually or runs after a successful ingestion workflow. It trains the
separate XGBoost profile only from eligible rows on or after
`2021-03-23 00:00:00`, uses `cape`, `freezing_level_height`, and `uv_index`,
logs to its own MLflow experiment, and uploads
`artifacts/xgboost-historical-forecast-training`.

Add these GitHub secrets:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

Optional repository variables:

```text
OPENMETEO_LAT
OPENMETEO_LON
OPENMETEO_LOCATION_NAME
SUPABASE_RAW_TABLE
SUPABASE_FEATURE_TABLE
```

## Generated Files

Generated data, model artifacts, MLflow runs, caches, and local CSV exports are ignored by git.
