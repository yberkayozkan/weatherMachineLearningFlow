from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities import Metric
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from weather_ml.openmeteo_features import (
    FORBIDDEN_FUTURE_TARGET_COLUMNS,
    WEATHER_CONDITION_LABELS,
)

load_dotenv(dotenv_path=Path(".env"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

DEFAULT_TARGET = "target_weather_condition_24h"
VALIDATION_GAP_HOURS = 24
MLFLOW_METRIC_BATCH_SIZE = 500
SUMMARY_METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "roc_auc_ovr_weighted",
    "log_loss",
]
HISTORY_METRIC_NAMES = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr_weighted",
    "log_loss",
]

DEFAULT_FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "dew_point_spread",
    "apparent_temperature",
    "rain",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_gusts_10m",
    "weather_code_lag_4h",
    "weather_code_lag_12h",
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
    "temperature_2m_max_lag_1d",
    "temperature_2m_min_lag_1d",
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
]
HISTORICAL_FORECAST_FEATURE_COLUMNS = [
    *DEFAULT_FEATURE_COLUMNS,
    "cape",
    "freezing_level_height",
    "uv_index",
]
HISTORICAL_FORECAST_TRAINING_START = "2021-03-23 00:00:00"

LEAKAGE_COLUMNS = [
    DEFAULT_TARGET,
    "weather_code",
    "weather_condition",
    *FORBIDDEN_FUTURE_TARGET_COLUMNS,
]


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path
    feature_importance_plot_path: Path
    metrics_history_path: Path
    metrics_history_plot_path: Path
    time_series_cv_metrics_path: Path
    time_series_cv_plot_path: Path
    roc_curve_path: Path | None


@dataclass(frozen=True)
class TemporalValidationSplit:
    final_train: pd.DataFrame
    final_test: pd.DataFrame
    cv_indices: list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class XGBoostTrainingParams:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    acceptance_threshold: float = 0.40


def train_weather_condition_model(
    frame: pd.DataFrame,
    *,
    output_dir: str = "artifacts/training",
    target_column: str = DEFAULT_TARGET,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    experiment_name: str = "weather-condition-xgboost",
    log_to_mlflow: bool = True,
    training_params: XGBoostTrainingParams | None = None,
    feature_columns: list[str] | None = None,
    model_filename: str = "weather_condition_xgboost_model.pkl",
    mlflow_run_name: str = "xgboost-weather-condition",
    training_scope: str = "full_history",
) -> TrainingArtifacts:
    selected_features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    _assert_training_contract(target_column, selected_features)
    params = training_params or XGBoostTrainingParams()
    data = _prepare_training_frame(
        frame,
        target_column=target_column,
        feature_columns=selected_features,
    )
    temporal_split = _build_temporal_validation_split(
        data, test_fraction=test_fraction, cv_splits=cv_splits
    )
    train = temporal_split.final_train
    test = temporal_split.final_test
    x_train = train[selected_features]
    y_train_raw = train[target_column].astype(str)
    x_test = test[selected_features]
    y_test_raw = test[target_column].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train_raw, y_test_raw], ignore_index=True))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    cv_metrics = _build_xgboost_time_series_cv_metrics(
        train=train,
        cv_indices=temporal_split.cv_indices,
        target_column=target_column,
        test_fraction=test_fraction,
        label_encoder=label_encoder,
        training_params=params,
        feature_columns=selected_features,
    )
    model = _build_xgboost_classifier(params, len(label_encoder.classes_))
    logger.info(
        "starting model training: rows_train=%s rows_test=%s n_estimators=%s",
        len(train),
        len(test),
        params.n_estimators,
    )
    model.fit(x_train, y_train, sample_weight=_build_balanced_sample_weight(y_train))
    logger.info("completed model training")

    predicted_labels = model.predict(x_test)
    predicted_conditions = label_encoder.inverse_transform(predicted_labels)
    probabilities = model.predict_proba(x_test)

    metrics = _build_metrics(
        y_test=y_test,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        label_encoder=label_encoder,
        test_fraction=test_fraction,
        train=train,
        test=test,
        target_column=target_column,
        acceptance_threshold=params.acceptance_threshold,
        model_params_name="xgboost_params",
        model_params={
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "reg_lambda": params.reg_lambda,
            "eval_metric": "mlogloss",
        },
        feature_columns=selected_features,
    )
    _add_time_series_validation_metrics(metrics, cv_metrics=cv_metrics, cv_splits=cv_splits)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / model_filename
    metrics_path = out / "metrics.json"
    metrics_history_path = out / "metrics_history.csv"
    predictions_path = out / "test_predictions.csv"
    confusion_matrix_path = out / "confusion_matrix.png"
    feature_importance_path = out / "feature_importance.csv"
    feature_importance_plot_path = out / "feature_importance.png"
    metrics_history_plot_path = out / "metrics_history.png"
    time_series_cv_metrics_path = out / "time_series_cv_metrics.csv"
    time_series_cv_plot_path = out / "time_series_cv_metrics.png"
    roc_curve_path = out / "roc_auc_ovr.png"
    legacy_metrics_plot_path = out / "metrics_summary.png"
    legacy_metrics_plot_path.unlink(missing_ok=True)

    logger.info("starting artifact generation: output_dir=%s", out)
    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": selected_features,
        "target_column": target_column,
        "weather_condition_labels": WEATHER_CONDITION_LABELS,
    }
    with model_path.open("wb") as file:
        pickle.dump(payload, file)

    feature_importance = _build_feature_importance(model, selected_features)
    metrics_history = _build_metrics_history(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        classes=label_encoder.classes_,
        total_iterations=params.n_estimators,
    )
    metrics["metrics_history_csv"] = str(metrics_history_path)
    metrics["feature_importance_csv"] = str(feature_importance_path)
    metrics["time_series_cv_metrics_csv"] = str(time_series_cv_metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_importance.to_csv(feature_importance_path, index=False)
    metrics_history.to_csv(metrics_history_path, index=False)
    cv_metrics.to_csv(time_series_cv_metrics_path, index=False)
    pd.DataFrame(
        {
            "observed_at": test["observed_at"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_weather_condition": y_test_raw.to_numpy(),
            "predicted_weather_condition": predicted_conditions,
            "prediction_confidence": probabilities.max(axis=1),
        }
    ).to_csv(predictions_path, index=False)
    _write_confusion_matrix(confusion_matrix_path, y_test, predicted_labels, label_encoder.classes_)
    _write_feature_importance_plot(feature_importance_plot_path, feature_importance)
    _write_metrics_history_plot(metrics_history_plot_path, metrics_history)
    _write_time_series_cv_plot(time_series_cv_plot_path, cv_metrics)
    roc_curve_written = _write_roc_curve(
        roc_curve_path,
        y_test,
        probabilities,
        label_encoder.classes_,
    )
    logger.info("completed artifact generation: output_dir=%s", out)

    if log_to_mlflow:
        logger.info("starting MLflow upload: experiment=%s", experiment_name)
        _log_mlflow_run(
            experiment_name=experiment_name,
            model=model,
            metrics=metrics,
            output_dir=out,
            label_encoder=label_encoder,
            training_params=params,
            metrics_history=metrics_history,
            cv_metrics=cv_metrics,
            cv_splits=cv_splits,
            mlflow_run_name=mlflow_run_name,
            feature_columns=selected_features,
            training_scope=training_scope,
        )

    return TrainingArtifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_matrix_path=confusion_matrix_path,
        feature_importance_path=feature_importance_path,
        feature_importance_plot_path=feature_importance_plot_path,
        metrics_history_path=metrics_history_path,
        metrics_history_plot_path=metrics_history_plot_path,
        time_series_cv_metrics_path=time_series_cv_metrics_path,
        time_series_cv_plot_path=time_series_cv_plot_path,
        roc_curve_path=roc_curve_path if roc_curve_written else None,
    )


def train_historical_forecast_xgboost_model(
    frame: pd.DataFrame,
    *,
    output_dir: str = "artifacts/xgboost-historical-forecast-training",
    target_column: str = DEFAULT_TARGET,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    experiment_name: str = "weather-condition-xgboost-historical-forecast",
    log_to_mlflow: bool = True,
    training_params: XGBoostTrainingParams | None = None,
) -> TrainingArtifacts:
    observed_at = pd.to_datetime(frame["observed_at"], errors="coerce")
    eligible = frame.loc[observed_at >= pd.Timestamp(HISTORICAL_FORECAST_TRAINING_START)].copy()
    if eligible.empty:
        raise ValueError(
            "No rows are available on or after "
            f"{HISTORICAL_FORECAST_TRAINING_START} for Historical Forecast training."
        )
    return train_weather_condition_model(
        eligible,
        output_dir=output_dir,
        target_column=target_column,
        test_fraction=test_fraction,
        cv_splits=cv_splits,
        experiment_name=experiment_name,
        log_to_mlflow=log_to_mlflow,
        training_params=training_params,
        feature_columns=HISTORICAL_FORECAST_FEATURE_COLUMNS,
        model_filename="weather_condition_xgboost_historical_forecast_model.pkl",
        mlflow_run_name="xgboost-historical-forecast-weather-condition",
        training_scope="historical_forecast_2021_03_23_plus",
    )


def _prepare_training_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(
            "Training dataframe is empty. Populate the Supabase feature table before training."
        )

    selected_features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    missing = [
        column
        for column in [*selected_features, target_column, "observed_at"]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing training columns: {', '.join(missing)}")

    data = frame.copy()
    data["observed_at"] = pd.to_datetime(data["observed_at"])

    data = data.sort_values("observed_at").dropna(subset=[*selected_features, target_column])
    data["is_weekend"] = data["is_weekend"].astype(bool).astype(int)
    data["weather_condition_lag_4h_code"] = data["weather_condition_lag_4h_code"].astype(int)
    data["weather_condition_lag_12h_code"] = data["weather_condition_lag_12h_code"].astype(int)
    return data


def _build_temporal_validation_split(
    data: pd.DataFrame, *, test_fraction: float, cv_splits: int
) -> TemporalValidationSplit:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2 for time-series validation.")

    split_index = int(len(data) * (1 - test_fraction))
    final_train_end = split_index - VALIDATION_GAP_HOURS
    if final_train_end <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows for final holdout with a fixed 24-hour validation gap.")

    final_train = data.iloc[:final_train_end].copy()
    final_test = data.iloc[split_index:].copy()
    try:
        cv_indices = list(
            TimeSeriesSplit(n_splits=cv_splits, gap=VALIDATION_GAP_HOURS).split(final_train)
        )
    except ValueError as exc:
        raise ValueError(
            "Not enough pre-holdout rows for time-series cross-validation "
            f"with {cv_splits} splits and a {VALIDATION_GAP_HOURS}-hour gap."
        ) from exc
    return TemporalValidationSplit(
        final_train=final_train,
        final_test=final_test,
        cv_indices=cv_indices,
    )


def _build_xgboost_classifier(params: XGBoostTrainingParams, class_count: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def _build_xgboost_time_series_cv_metrics(
    *,
    train: pd.DataFrame,
    cv_indices: list[tuple[np.ndarray, np.ndarray]],
    target_column: str,
    test_fraction: float,
    label_encoder: LabelEncoder,
    training_params: XGBoostTrainingParams,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(cv_indices, start=1):
        fold_train = train.iloc[train_indices]
        validation = train.iloc[validation_indices]
        model = _build_xgboost_classifier(training_params, len(label_encoder.classes_))
        model.fit(
            fold_train[feature_columns],
            label_encoder.transform(fold_train[target_column].astype(str)),
            sample_weight=_build_balanced_sample_weight(
                label_encoder.transform(fold_train[target_column].astype(str))
            ),
        )
        y_validation = label_encoder.transform(validation[target_column].astype(str))
        probabilities = model.predict_proba(validation[feature_columns])
        predicted_labels = probabilities.argmax(axis=1)
        fold_metrics = _build_metrics(
            y_test=y_validation,
            predicted_labels=predicted_labels,
            probabilities=probabilities,
            label_encoder=label_encoder,
            test_fraction=test_fraction,
            train=fold_train,
            test=validation,
            target_column=target_column,
            acceptance_threshold=training_params.acceptance_threshold,
            model_params_name="xgboost_params",
            model_params={},
            feature_columns=feature_columns,
        )
        rows.append(_build_cv_fold_row(fold, fold_train, validation, fold_metrics))
    return pd.DataFrame(rows)


def _build_balanced_sample_weight(y_train: np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y_train)


def _build_cv_fold_row(
    fold: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fold_metrics: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "fold": fold,
        "train_start": str(train["observed_at"].iloc[0]),
        "train_end": str(train["observed_at"].iloc[-1]),
        "validation_start": str(validation["observed_at"].iloc[0]),
        "validation_end": str(validation["observed_at"].iloc[-1]),
        "rows_train": len(train),
        "rows_validation": len(validation),
        "gap_hours": VALIDATION_GAP_HOURS,
    }
    row.update({key: fold_metrics[key] for key in SUMMARY_METRIC_NAMES})
    return row


def _add_time_series_validation_metrics(
    metrics: dict[str, object], *, cv_metrics: pd.DataFrame, cv_splits: int
) -> None:
    metrics["validation_strategy"] = "time_series_split"
    metrics["cv_splits"] = cv_splits
    metrics["validation_gap_hours"] = VALIDATION_GAP_HOURS
    metrics["cv_mean_metrics"] = {
        key: None if cv_metrics[key].dropna().empty else float(cv_metrics[key].mean())
        for key in SUMMARY_METRIC_NAMES
    }


def _build_metrics(
    *,
    y_test: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
    test_fraction: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    acceptance_threshold: float,
    model_params_name: str,
    model_params: dict[str, object],
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    classes = list(label_encoder.classes_)
    roc_auc = _safe_multiclass_roc_auc(y_test, probabilities, len(classes))
    f1_weighted = float(f1_score(y_test, predicted_labels, average="weighted", zero_division=0))
    precision_weighted = float(
        precision_score(y_test, predicted_labels, average="weighted", zero_division=0)
    )
    loss = _safe_log_loss(y_test, probabilities, len(classes))
    accepted = f1_weighted >= acceptance_threshold
    return {
        "rows_total": int(len(train) + len(test)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target_column": target_column,
        "test_fraction": test_fraction,
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_labels)),
        "precision_weighted": precision_weighted,
        "recall_macro": float(
            recall_score(y_test, predicted_labels, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predicted_labels, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "f1_weighted": f1_weighted,
        "roc_auc_ovr_weighted": roc_auc,
        "log_loss": loss,
        "accepted": accepted,
        "acceptance_metric": "f1_weighted",
        "acceptance_threshold": acceptance_threshold,
        model_params_name: model_params,
        "classification_report": classification_report(
            y_test,
            predicted_labels,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(
                y_test, predicted_labels, labels=list(range(len(classes)))
            ).tolist(),
        },
        "train_start": str(train["observed_at"].iloc[0]),
        "train_end": str(train["observed_at"].iloc[-1]),
        "test_start": str(test["observed_at"].iloc[0]),
        "test_end": str(test["observed_at"].iloc[-1]),
        "feature_columns": list(feature_columns or DEFAULT_FEATURE_COLUMNS),
    }


def _build_metrics_history(
    *,
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: np.ndarray,
    total_iterations: int,
) -> pd.DataFrame:
    step = max(1, total_iterations // 100)
    iterations = list(range(step, total_iterations + 1, step))
    if iterations[-1] != total_iterations:
        iterations.append(total_iterations)

    rows: list[dict[str, object]] = []
    for iteration in iterations:
        for split_name, x_values, y_values in [
            ("train", x_train, y_train),
            ("test", x_test, y_test),
        ]:
            probabilities = model.predict_proba(x_values, iteration_range=(0, iteration))
            predicted_labels = probabilities.argmax(axis=1)
            rows.append(
                {
                    "iteration": iteration,
                    "split": split_name,
                    "accuracy": float(accuracy_score(y_values, predicted_labels)),
                    "precision_weighted": float(
                        precision_score(
                            y_values, predicted_labels, average="weighted", zero_division=0
                        )
                    ),
                    "recall_weighted": float(
                        recall_score(
                            y_values, predicted_labels, average="weighted", zero_division=0
                        )
                    ),
                    "f1_weighted": float(
                        f1_score(y_values, predicted_labels, average="weighted", zero_division=0)
                    ),
                    "roc_auc_ovr_weighted": _safe_multiclass_roc_auc(
                        y_values, probabilities, len(classes)
                    ),
                    "log_loss": _safe_log_loss(y_values, probabilities, len(classes)),
                }
            )
    return pd.DataFrame(rows)


def _build_feature_importance(
    model: XGBClassifier,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": list(feature_columns or DEFAULT_FEATURE_COLUMNS),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _safe_multiclass_roc_auc(
    y_test: np.ndarray, probabilities: np.ndarray, class_count: int
) -> float | None:
    present_classes = sorted(set(y_test.tolist()))
    if len(present_classes) < 2:
        return None
    try:
        y_test_binarized = label_binarize(y_test, classes=present_classes)
        present_probabilities = probabilities[:, present_classes]
        return float(
            roc_auc_score(
                y_test_binarized,
                present_probabilities,
                average="weighted",
                multi_class="ovr",
            )
        )
    except ValueError:
        return None


def _safe_log_loss(y_test: np.ndarray, probabilities: np.ndarray, class_count: int) -> float | None:
    try:
        return float(log_loss(y_test, probabilities, labels=list(range(class_count))))
    except ValueError:
        return None


def _write_confusion_matrix(
    output_path: Path,
    y_test: np.ndarray,
    predicted_labels: np.ndarray,
    labels: np.ndarray,
) -> None:
    matrix = confusion_matrix(y_test, predicted_labels, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_feature_importance_plot(
    output_path: Path,
    feature_importance: pd.DataFrame,
    *,
    model_name: str = "XGBoost",
) -> None:
    ordered = feature_importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(ordered["feature"], ordered["importance"], color="#2563eb")
    ax.set_xlabel("importance")
    ax.set_title(f"{model_name} feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_metrics_history_plot(output_path: Path, history: pd.DataFrame) -> None:
    metric_names = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_ovr_weighted",
        "log_loss",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, metric_name in zip(axes.flatten(), metric_names, strict=True):
        for split_name, split_frame in history.groupby("split"):
            ax.plot(
                split_frame["iteration"],
                split_frame[metric_name],
                label=split_name,
                linewidth=1.8,
            )
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("boosting iteration")
        ax.grid(True, alpha=0.25)
        if metric_name != "log_loss":
            ax.set_ylim(0, 1)
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_time_series_cv_plot(output_path: Path, cv_metrics: pd.DataFrame) -> None:
    metric_names = ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric_name in metric_names:
        if cv_metrics[metric_name].notna().any():
            ax.plot(
                cv_metrics["fold"],
                cv_metrics[metric_name],
                marker="o",
                linewidth=1.8,
                label=metric_name,
            )
    ax.set_xlabel("validation fold")
    ax.set_ylabel("score")
    ax.set_title("Time-series cross-validation scores")
    ax.set_xticks(cv_metrics["fold"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_roc_curve(
    output_path: Path,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> bool:
    from sklearn.metrics import auc, roc_curve

    output_path.unlink(missing_ok=True)
    present_classes = sorted(set(y_test.tolist()))
    if len(present_classes) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    wrote_curve = False
    for class_index in present_classes:
        binary_target = (y_test == class_index).astype(int)
        if len(set(binary_target.tolist())) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_target,
            probabilities[:, class_index],
        )
        curve_auc = auc(false_positive_rate, true_positive_rate)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{labels[class_index]} AUC={curve_auc:.3f}",
        )
        wrote_curve = True

    if not wrote_curve:
        plt.close(fig)
        return False

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _log_mlflow_run(
    *,
    experiment_name: str,
    model: XGBClassifier,
    metrics: dict[str, object],
    output_dir: Path,
    label_encoder: LabelEncoder,
    training_params: XGBoostTrainingParams,
    metrics_history: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    cv_splits: int,
    mlflow_run_name: str = "xgboost-weather-condition",
    feature_columns: list[str] | None = None,
    training_scope: str = "full_history",
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "target_column": metrics["target_column"],
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
                "reg_lambda": training_params.reg_lambda,
                "acceptance_threshold": training_params.acceptance_threshold,
                "classes": ",".join(label_encoder.classes_),
                "validation_strategy": "time_series_split",
                "cv_splits": cv_splits,
                "validation_gap_hours": VALIDATION_GAP_HOURS,
                "class_weighting": "balanced_sample_weight",
                "training_scope": training_scope,
                "feature_columns": ",".join(feature_columns or DEFAULT_FEATURE_COLUMNS),
            }
        )
        mlflow.set_tags(
            {
                "model_status": "accepted" if metrics["accepted"] else "rejected",
                "acceptance_metric": metrics["acceptance_metric"],
                "acceptance_threshold": str(metrics["acceptance_threshold"]),
            }
        )
        summary_metrics = {
            key: float(metrics[key]) for key in SUMMARY_METRIC_NAMES if metrics[key] is not None
        }
        summary_metrics.update(
            {
                f"cv_mean_{key}": float(value)
                for key, value in metrics["cv_mean_metrics"].items()
                if value is not None
            }
        )
        mlflow.log_metrics(summary_metrics)

        history_metrics = [
            *_build_mlflow_history_metrics(metrics_history),
            *_build_mlflow_cv_metrics(cv_metrics),
        ]
        _log_mlflow_metric_batches(run.info.run_id, history_metrics)
        mlflow.log_artifacts(str(output_dir))
    logger.info("completed MLflow upload: history_metrics=%s", len(history_metrics))


def _build_mlflow_history_metrics(metrics_history: pd.DataFrame) -> list[Metric]:
    timestamp = int(time.time() * 1000)
    history_metrics: list[Metric] = []
    for row in metrics_history.to_dict(orient="records"):
        step = int(row["iteration"])
        split = str(row["split"])
        for key in HISTORY_METRIC_NAMES:
            value = row[key]
            if pd.isna(value):
                continue
            history_metrics.append(
                Metric(
                    key=f"{split}_{key}",
                    value=float(value),
                    timestamp=timestamp,
                    step=step,
                )
            )
    return history_metrics


def _build_mlflow_cv_metrics(cv_metrics: pd.DataFrame) -> list[Metric]:
    timestamp = int(time.time() * 1000)
    metrics: list[Metric] = []
    for row in cv_metrics.to_dict(orient="records"):
        step = int(row["fold"])
        for key in SUMMARY_METRIC_NAMES:
            value = row[key]
            if pd.isna(value):
                continue
            metrics.append(
                Metric(
                    key=f"cv_validation_{key}",
                    value=float(value),
                    timestamp=timestamp,
                    step=step,
                )
            )
    return metrics


def _log_mlflow_metric_batches(run_id: str, metrics: list[Metric]) -> None:
    client = MlflowClient()
    for start in range(0, len(metrics), MLFLOW_METRIC_BATCH_SIZE):
        client.log_batch(
            run_id,
            metrics=metrics[start : start + MLFLOW_METRIC_BATCH_SIZE],
        )


train_weather_code_model = train_weather_condition_model
train_temperature_model = train_weather_condition_model


def _assert_training_contract(
    target_column: str,
    feature_columns: list[str] | None = None,
) -> None:
    if target_column != DEFAULT_TARGET:
        raise ValueError(f"Weather condition training target must be {DEFAULT_TARGET}.")

    selected_features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    leaked = [column for column in LEAKAGE_COLUMNS if column in selected_features]
    if leaked:
        raise ValueError(f"Training feature columns include leakage columns: {', '.join(leaked)}")
