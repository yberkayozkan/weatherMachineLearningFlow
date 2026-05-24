# Weather Machine Learning

Open-Meteo hourly archive ingestion, Supabase loading, and XGBoost weather-condition training.

## Pipeline

1. Download Open-Meteo hourly archive data.
2. Build lag, rolling, calendar, and weather-condition features.
3. Write only target-safe rows to Supabase.
4. Train an XGBoost classifier for `target_weather_condition_24h` using current and historical features.
5. Log metrics and artifacts to DagsHub MLflow.

The model predicts one of these weather-condition groups 24 hours ahead. The only
future label stored is `target_weather_condition_24h`; numeric future targets and
future weather code targets are not stored.

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
MLFLOW_TRACKING_URI=https://dagshub.com/dragnellstr/mlflowdeneme.mlflow
MLFLOW_TRACKING_USERNAME=dragnellstr
MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

## Supabase Tables

Create the canonical tables with:

```powershell
psql "$env:SUPABASE_DB_URL" -f sql/005_kadikoy_weather_code_tables.sql
```

Default table names:

```text
KadikoyWeatherCodeRaw
KadikoyWeatherCodeFeature
```

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
duplicates.

Train from Supabase and log to MLflow:

```powershell
python -m weather_ml.pipelines.train_from_supabase --table-name KadikoyWeatherCodeFeature
```

Train from a local feature CSV without MLflow:

```powershell
python -m weather_ml.pipelines.train_from_csv --csv-path path\to\features.csv --no-mlflow
```

## GitHub Actions

`.github/workflows/weather-ingestion.yml` runs the Open-Meteo to Supabase pipeline daily.

`.github/workflows/train-model.yml` runs after ingestion succeeds and uploads:

```text
weather_condition_xgboost_model.pkl
metrics.json
metrics_history.csv
test_predictions.csv
confusion_matrix.png
metrics_summary.png
metrics_history.png
roc_auc_ovr.png
```

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
