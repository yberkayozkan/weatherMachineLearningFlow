# Weather Machine Learning

An hourly weather data pipeline that retrieves Open-Meteo archive observations,
engineers time-based features, persists raw and feature-ready records in
Supabase, and trains XGBoost or LightGBM classifiers to predict the weather
condition 24 hours ahead.

## Pipeline

1. Retrieve hourly archive observations from Open-Meteo.
2. Map WMO weather codes into condition groups and build lag, rolling-window,
   and calendar features.
3. Upsert raw observations and target-safe feature rows into Supabase tables
   keyed by `observed_at`.
4. Train an `XGBClassifier` or `LGBMClassifier` for the multiclass
   `target_weather_condition_24h` target using chronological train/test
   splitting.
5. Manually log model parameters, summary metrics, run tags, iteration-level
   train/test metrics, and generated artifacts to MLflow.

The only future label stored in the feature table is
`target_weather_condition_24h`; numeric future targets and future weather-code
targets are excluded.

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

Train LightGBM from Supabase and log to its MLflow experiment:

```powershell
python -m weather_ml.pipelines.train_lightgbm_from_supabase --table-name KadikoyWeatherCodeFeature
```

Train LightGBM from a local feature CSV without MLflow:

```powershell
python -m weather_ml.pipelines.train_lightgbm_from_csv --csv-path path\to\features.csv --no-mlflow
```

## Training Outputs

XGBoost writes to `artifacts/training` by default with model file
`weather_condition_xgboost_model.pkl`. LightGBM writes to
`artifacts/lightgbm-training` by default with model file
`weather_condition_lightgbm_model.pkl`. Each training run also writes the
following shared outputs; when MLflow logging is enabled, the output directory
is uploaded to the run.

```text
metrics.json
metrics_history.csv
test_predictions.csv
confusion_matrix.png
feature_importance.csv
feature_importance.png
metrics_history.png
roc_auc_ovr.png (when ROC curves can be calculated)
```

`feature_importance.csv` and `feature_importance.png` report the trained
model's `feature_importances_` values. XGBoost logs to the default experiment
`weather-condition-xgboost`; LightGBM logs to `weather-condition-lightgbm`.
MLflow metrics are logged explicitly by the training code; MLflow autologging
is not enabled.

## GitHub Actions

`.github/workflows/weather-ingestion.yml` runs on a daily schedule and can be
started manually. It downloads a recent Open-Meteo window, builds features,
and writes raw and feature rows to Supabase.

`.github/workflows/train-model.yml` can be started manually with XGBoost and
split parameters, or runs after a successful ingestion workflow. It trains
from the Supabase feature table, logs to MLflow, and uploads the training
output directory as a GitHub Actions artifact.

`.github/workflows/train-lightgbm-model.yml` can be started manually with
LightGBM and split parameters, or runs after a successful ingestion workflow.
It trains from the Supabase feature table, logs to the LightGBM MLflow
experiment, and uploads `artifacts/lightgbm-training`.

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
