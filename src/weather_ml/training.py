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
from sklearn.preprocessing import LabelEncoder, label_binarize
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
    "apparent_temperature",
    "rain",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
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
    metrics_plot_path: Path
    metrics_history_path: Path
    metrics_history_plot_path: Path
    roc_curve_path: Path | None


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
    experiment_name: str = "weather-condition-xgboost",
    log_to_mlflow: bool = True,
    training_params: XGBoostTrainingParams | None = None,
) -> TrainingArtifacts:
    _assert_training_contract(target_column)
    params = training_params or XGBoostTrainingParams()
    data = _prepare_training_frame(frame, target_column=target_column)

    split_index = int(len(data) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows for train/test split.")

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    x_train = train[DEFAULT_FEATURE_COLUMNS]
    y_train_raw = train[target_column].astype(str)
    x_test = test[DEFAULT_FEATURE_COLUMNS]
    y_test_raw = test[target_column].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train_raw, y_test_raw], ignore_index=True))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
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
    logger.info(
        "starting model training: rows_train=%s rows_test=%s n_estimators=%s",
        len(train),
        len(test),
        params.n_estimators,
    )
    model.fit(x_train, y_train)
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
        training_params=params,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "weather_condition_xgboost_model.pkl"
    metrics_path = out / "metrics.json"
    metrics_history_path = out / "metrics_history.csv"
    predictions_path = out / "test_predictions.csv"
    confusion_matrix_path = out / "confusion_matrix.png"
    metrics_plot_path = out / "metrics_summary.png"
    metrics_history_plot_path = out / "metrics_history.png"
    roc_curve_path = out / "roc_auc_ovr.png"

    logger.info("starting artifact generation: output_dir=%s", out)
    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "target_column": target_column,
        "weather_condition_labels": WEATHER_CONDITION_LABELS,
    }
    with model_path.open("wb") as file:
        pickle.dump(payload, file)

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
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_history.to_csv(metrics_history_path, index=False)
    pd.DataFrame(
        {
            "observed_at": test["observed_at"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_weather_condition": y_test_raw.to_numpy(),
            "predicted_weather_condition": predicted_conditions,
            "prediction_confidence": probabilities.max(axis=1),
        }
    ).to_csv(predictions_path, index=False)
    _write_confusion_matrix(confusion_matrix_path, y_test, predicted_labels, label_encoder.classes_)
    _write_metrics_plot(metrics_plot_path, metrics)
    _write_metrics_history_plot(metrics_history_plot_path, metrics_history)
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
        )

    return TrainingArtifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_matrix_path=confusion_matrix_path,
        metrics_plot_path=metrics_plot_path,
        metrics_history_path=metrics_history_path,
        metrics_history_plot_path=metrics_history_plot_path,
        roc_curve_path=roc_curve_path if roc_curve_written else None,
    )


def _prepare_training_frame(frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Training dataframe is empty. Populate the Supabase feature table before training.")

    missing = [column for column in [*DEFAULT_FEATURE_COLUMNS, target_column, "observed_at"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing training columns: {', '.join(missing)}")

    data = frame.copy()
    data["observed_at"] = pd.to_datetime(data["observed_at"])

    data = data.sort_values("observed_at").dropna(subset=[*DEFAULT_FEATURE_COLUMNS, target_column])
    data["is_weekend"] = data["is_weekend"].astype(bool).astype(int)
    data["weather_condition_lag_4h_code"] = data["weather_condition_lag_4h_code"].astype(int)
    data["weather_condition_lag_12h_code"] = data["weather_condition_lag_12h_code"].astype(int)
    return data


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
    training_params: XGBoostTrainingParams,
) -> dict[str, object]:
    classes = list(label_encoder.classes_)
    roc_auc = _safe_multiclass_roc_auc(y_test, probabilities, len(classes))
    f1_weighted = float(f1_score(y_test, predicted_labels, average="weighted", zero_division=0))
    precision_weighted = float(precision_score(y_test, predicted_labels, average="weighted", zero_division=0))
    loss = _safe_log_loss(y_test, probabilities, len(classes))
    accepted = f1_weighted >= training_params.acceptance_threshold
    return {
        "rows_total": int(len(train) + len(test)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target_column": target_column,
        "test_fraction": test_fraction,
        "accuracy": float(accuracy_score(y_test, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_labels)),
        "precision_weighted": precision_weighted,
        "recall_macro": float(recall_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predicted_labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "f1_weighted": f1_weighted,
        "roc_auc_ovr_weighted": roc_auc,
        "log_loss": loss,
        "accepted": accepted,
        "acceptance_metric": "f1_weighted",
        "acceptance_threshold": training_params.acceptance_threshold,
        "xgboost_params": {
            "n_estimators": training_params.n_estimators,
            "max_depth": training_params.max_depth,
            "learning_rate": training_params.learning_rate,
            "subsample": training_params.subsample,
            "colsample_bytree": training_params.colsample_bytree,
            "reg_lambda": training_params.reg_lambda,
            "eval_metric": "mlogloss",
        },
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
            "matrix": confusion_matrix(y_test, predicted_labels, labels=list(range(len(classes)))).tolist(),
        },
        "train_start": str(train["observed_at"].iloc[0]),
        "train_end": str(train["observed_at"].iloc[-1]),
        "test_start": str(test["observed_at"].iloc[0]),
        "test_end": str(test["observed_at"].iloc[-1]),
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
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
                        precision_score(y_values, predicted_labels, average="weighted", zero_division=0)
                    ),
                    "recall_weighted": float(
                        recall_score(y_values, predicted_labels, average="weighted", zero_division=0)
                    ),
                    "f1_weighted": float(f1_score(y_values, predicted_labels, average="weighted", zero_division=0)),
                    "roc_auc_ovr_weighted": _safe_multiclass_roc_auc(y_values, probabilities, len(classes)),
                    "log_loss": _safe_log_loss(y_values, probabilities, len(classes)),
                }
            )
    return pd.DataFrame(rows)


def _safe_multiclass_roc_auc(y_test: np.ndarray, probabilities: np.ndarray, class_count: int) -> float | None:
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


def _write_metrics_plot(output_path: Path, metrics: dict[str, object]) -> None:
    metric_names = [
        "accuracy",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_weighted",
        "log_loss",
    ]
    values = [metrics[name] for name in metric_names]
    labels = [name.replace("_", " ") for name in metric_names]
    numeric_values = [0 if value is None else float(value) for value in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, numeric_values, color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Weather condition model metrics")
    ax.tick_params(axis="x", rotation=30)
    for index, value in enumerate(values):
        text = "n/a" if value is None else f"{float(value):.3f}"
        ax.text(index, numeric_values[index] + 0.02, text, ha="center", fontsize=9)
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


def _write_roc_curve(
    output_path: Path,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> bool:
    from sklearn.metrics import auc, roc_curve

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
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost-weather-condition") as run:
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
            key: float(metrics[key])
            for key in SUMMARY_METRIC_NAMES
            if metrics[key] is not None
        }
        mlflow.log_metrics(summary_metrics)

        history_metrics = _build_mlflow_history_metrics(metrics_history)
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


def _log_mlflow_metric_batches(run_id: str, metrics: list[Metric]) -> None:
    client = MlflowClient()
    for start in range(0, len(metrics), MLFLOW_METRIC_BATCH_SIZE):
        client.log_batch(
            run_id,
            metrics=metrics[start : start + MLFLOW_METRIC_BATCH_SIZE],
        )


train_weather_code_model = train_weather_condition_model
train_temperature_model = train_weather_condition_model


def _assert_training_contract(target_column: str) -> None:
    if target_column != DEFAULT_TARGET:
        raise ValueError(f"Weather condition training target must be {DEFAULT_TARGET}.")

    leaked = [column for column in LEAKAGE_COLUMNS if column in DEFAULT_FEATURE_COLUMNS]
    if leaked:
        raise ValueError(f"Training feature columns include leakage columns: {', '.join(leaked)}")
