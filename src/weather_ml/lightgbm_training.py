from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from weather_ml.openmeteo_features import WEATHER_CONDITION_LABELS
from weather_ml.training import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET,
    SUMMARY_METRIC_NAMES,
    VALIDATION_GAP_HOURS,
    TrainingArtifacts,
    _add_time_series_validation_metrics,
    _assert_training_contract,
    _build_cv_fold_row,
    _build_feature_importance,
    _build_metrics,
    _build_mlflow_cv_metrics,
    _build_mlflow_history_metrics,
    _build_sqrt_balanced_sample_weight,
    _build_temporal_validation_split,
    _log_mlflow_metric_batches,
    _log_registered_pyfunc_model,
    _prepare_training_frame,
    _safe_log_loss,
    _safe_multiclass_roc_auc,
    _write_confusion_matrix,
    _write_feature_importance_plot,
    _write_metrics_history_plot,
    _write_roc_curve,
    _write_time_series_cv_plot,
)

logger = logging.getLogger(__name__)
LIGHTGBM_REGISTERED_MODEL_NAME = "weather-condition-lightgbm"


@dataclass(frozen=True)
class LightGBMTrainingParams:
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    acceptance_threshold: float = 0.40


def train_lightgbm_weather_condition_model(
    frame: pd.DataFrame,
    *,
    output_dir: str = "artifacts/lightgbm-training",
    target_column: str = DEFAULT_TARGET,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    experiment_name: str = "weather-condition-lightgbm",
    log_to_mlflow: bool = True,
    training_params: LightGBMTrainingParams | None = None,
) -> TrainingArtifacts:
    _assert_training_contract(target_column)
    params = training_params or LightGBMTrainingParams()
    data = _prepare_training_frame(frame, target_column=target_column)
    temporal_split = _build_temporal_validation_split(
        data, test_fraction=test_fraction, cv_splits=cv_splits
    )
    train = temporal_split.final_train
    test = temporal_split.final_test
    x_train = train[DEFAULT_FEATURE_COLUMNS]
    y_train_raw = train[target_column].astype(str)
    x_test = test[DEFAULT_FEATURE_COLUMNS]
    y_test_raw = test[target_column].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train_raw, y_test_raw], ignore_index=True))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    cv_metrics = _build_lightgbm_time_series_cv_metrics(
        train=train,
        cv_indices=temporal_split.cv_indices,
        target_column=target_column,
        test_fraction=test_fraction,
        label_encoder=label_encoder,
        training_params=params,
    )
    model = _build_lightgbm_classifier(params, len(label_encoder.classes_))
    logger.info(
        "starting LightGBM training: rows_train=%s rows_test=%s n_estimators=%s",
        len(train),
        len(test),
        params.n_estimators,
    )
    model.fit(x_train, y_train, sample_weight=_build_sqrt_balanced_sample_weight(y_train))
    predicted_labels = model.predict(x_test).astype(int)
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
        model_params_name="lightgbm_params",
        model_params={
            "n_estimators": params.n_estimators,
            "learning_rate": params.learning_rate,
            "max_depth": params.max_depth,
            "num_leaves": params.num_leaves,
            "subsample": params.subsample,
            "subsample_freq": 1,
            "colsample_bytree": params.colsample_bytree,
            "reg_lambda": params.reg_lambda,
            "objective": "multiclass",
        },
    )
    _add_time_series_validation_metrics(metrics, cv_metrics=cv_metrics, cv_splits=cv_splits)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "weather_condition_lightgbm_model.pkl"
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

    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "label_encoder": label_encoder,
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "target_column": target_column,
                "weather_condition_labels": WEATHER_CONDITION_LABELS,
            },
            file,
        )

    feature_importance = _build_feature_importance(model)
    metrics_history = _build_lightgbm_metrics_history(
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
    _write_feature_importance_plot(
        feature_importance_plot_path, feature_importance, model_name="LightGBM"
    )
    _write_metrics_history_plot(metrics_history_plot_path, metrics_history)
    _write_time_series_cv_plot(time_series_cv_plot_path, cv_metrics)
    roc_curve_written = _write_roc_curve(
        roc_curve_path, y_test, probabilities, label_encoder.classes_
    )

    if log_to_mlflow:
        _log_lightgbm_mlflow_run(
            experiment_name=experiment_name,
            model=model,
            metrics=metrics,
            output_dir=out,
            label_encoder=label_encoder,
            training_params=params,
            metrics_history=metrics_history,
            cv_metrics=cv_metrics,
            cv_splits=cv_splits,
            model_path=model_path,
            registered_model_name=LIGHTGBM_REGISTERED_MODEL_NAME,
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


def _build_lightgbm_classifier(
    params: LightGBMTrainingParams, class_count: int
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=class_count,
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        num_leaves=params.num_leaves,
        subsample=params.subsample,
        subsample_freq=1,
        colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _build_lightgbm_time_series_cv_metrics(
    *,
    train: pd.DataFrame,
    cv_indices: list[tuple[np.ndarray, np.ndarray]],
    target_column: str,
    test_fraction: float,
    label_encoder: LabelEncoder,
    training_params: LightGBMTrainingParams,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(cv_indices, start=1):
        fold_train = train.iloc[train_indices]
        validation = train.iloc[validation_indices]
        model = _build_lightgbm_classifier(training_params, len(label_encoder.classes_))
        model.fit(
            fold_train[DEFAULT_FEATURE_COLUMNS],
            label_encoder.transform(fold_train[target_column].astype(str)),
            sample_weight=_build_sqrt_balanced_sample_weight(
                label_encoder.transform(fold_train[target_column].astype(str))
            ),
        )
        y_validation = label_encoder.transform(validation[target_column].astype(str))
        probabilities = model.predict_proba(validation[DEFAULT_FEATURE_COLUMNS])
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
            model_params_name="lightgbm_params",
            model_params={},
        )
        rows.append(_build_cv_fold_row(fold, fold_train, validation, fold_metrics))
    return pd.DataFrame(rows)


def _build_lightgbm_metrics_history(
    *,
    model: lgb.LGBMClassifier,
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
            probabilities = model.predict_proba(x_values, num_iteration=iteration)
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
                    "f1_macro": float(
                        f1_score(y_values, predicted_labels, average="macro", zero_division=0)
                    ),
                    "roc_auc_ovr_weighted": _safe_multiclass_roc_auc(
                        y_values, probabilities, len(classes)
                    ),
                    "log_loss": _safe_log_loss(y_values, probabilities, len(classes)),
                }
            )
    return pd.DataFrame(rows)


def _log_lightgbm_mlflow_run(
    *,
    experiment_name: str,
    model: lgb.LGBMClassifier,
    metrics: dict[str, object],
    output_dir: Path,
    label_encoder: LabelEncoder,
    training_params: LightGBMTrainingParams,
    metrics_history: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    cv_splits: int,
    model_path: Path | None = None,
    registered_model_name: str | None = LIGHTGBM_REGISTERED_MODEL_NAME,
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="lightgbm-weather-condition") as run:
        mlflow.log_params(
            {
                "model_type": "LGBMClassifier",
                "target_column": metrics["target_column"],
                "n_estimators": model.n_estimators,
                "learning_rate": model.learning_rate,
                "max_depth": model.max_depth,
                "num_leaves": model.num_leaves,
                "subsample": model.subsample,
                "subsample_freq": model.subsample_freq,
                "colsample_bytree": model.colsample_bytree,
                "reg_lambda": training_params.reg_lambda,
                "acceptance_threshold": training_params.acceptance_threshold,
                "classes": ",".join(label_encoder.classes_),
                "validation_strategy": "time_series_split",
                "cv_splits": cv_splits,
                "validation_gap_hours": VALIDATION_GAP_HOURS,
                "class_weighting": "sqrt_balanced_sample_weight",
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
        if model_path is not None and registered_model_name:
            _log_registered_pyfunc_model(
                model_path=model_path,
                registered_model_name=registered_model_name,
            )
    logger.info("completed LightGBM MLflow upload: history_metrics=%s", len(history_metrics))
