from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from weather_ml.openmeteo_features import WEATHER_CONDITION_LABELS
from weather_ml.training import (
    DEFAULT_TARGET,
    HISTORICAL_FORECAST_FEATURE_COLUMNS,
    HISTORICAL_FORECAST_TRAINING_START,
    SUMMARY_METRIC_NAMES,
    TrainingArtifacts,
    _assert_training_contract,
    _build_metrics,
    _log_mlflow_metric_batches,
    _prepare_training_frame,
    _write_confusion_matrix,
    _write_metrics_history_plot,
    _write_roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LSTMTrainingParams:
    sequence_length: int = 24
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 0.001
    acceptance_threshold: float = 0.40


class WeatherLSTM(nn.Module):
    def __init__(self, feature_count: int, class_count: int, params: LSTMTrainingParams) -> None:
        super().__init__()
        dropout = params.dropout if params.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Linear(params.hidden_size, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(values)
        return self.output(sequence_output[:, -1, :])


def train_lstm_weather_condition_model(
    frame: pd.DataFrame,
    *,
    output_dir: str = "artifacts/lstm-training",
    target_column: str = DEFAULT_TARGET,
    test_fraction: float = 0.2,
    experiment_name: str = "weather-condition-lstm-historical-forecast",
    log_to_mlflow: bool = True,
    training_params: LSTMTrainingParams | None = None,
) -> TrainingArtifacts:
    params = training_params or LSTMTrainingParams()
    feature_columns = list(HISTORICAL_FORECAST_FEATURE_COLUMNS)
    _assert_training_contract(target_column, feature_columns)
    observed_at = pd.to_datetime(frame["observed_at"], errors="coerce")
    eligible = frame.loc[observed_at >= pd.Timestamp(HISTORICAL_FORECAST_TRAINING_START)].copy()
    data = _prepare_training_frame(
        eligible,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    sequences, labels, sequence_times = _build_lstm_sequences(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=params.sequence_length,
    )
    split_index = int(len(sequences) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(sequences):
        raise ValueError("Not enough sequence rows for final LSTM holdout.")

    x_train_raw, x_test_raw = sequences[:split_index], sequences[split_index:]
    y_train_raw, y_test_raw = labels[:split_index], labels[split_index:]
    train_times = sequence_times[:split_index]
    test_times = sequence_times[split_index:]

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(
        x_train_raw.shape
    )
    x_test = scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(
        x_test_raw.shape
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = WeatherLSTM(len(feature_columns), len(label_encoder.classes_), params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=params.batch_size,
        shuffle=False,
    )
    history = _fit_lstm(
        model=model,
        train_loader=train_loader,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=params.epochs,
        device=device,
    )
    probabilities = _predict_lstm_probabilities(model, x_test, device)
    predicted_labels = probabilities.argmax(axis=1)
    predicted_conditions = label_encoder.inverse_transform(predicted_labels)

    train_metrics_frame = pd.DataFrame(
        {"observed_at": train_times, target_column: y_train_raw}
    )
    test_metrics_frame = pd.DataFrame({"observed_at": test_times, target_column: y_test_raw})
    metrics = _build_metrics(
        y_test=y_test,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        label_encoder=label_encoder,
        test_fraction=test_fraction,
        train=train_metrics_frame,
        test=test_metrics_frame,
        target_column=target_column,
        acceptance_threshold=params.acceptance_threshold,
        model_params_name="lstm_params",
        model_params={
            "sequence_length": params.sequence_length,
            "hidden_size": params.hidden_size,
            "num_layers": params.num_layers,
            "dropout": params.dropout,
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
        },
        feature_columns=feature_columns,
    )
    metrics["validation_strategy"] = "temporal_holdout_sequences"
    metrics["training_scope"] = "historical_forecast_2021_03_23_plus"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "weather_condition_lstm_historical_forecast_model.pkl"
    metrics_path = out / "metrics.json"
    predictions_path = out / "test_predictions.csv"
    confusion_matrix_path = out / "confusion_matrix.png"
    feature_importance_path = out / "feature_importance.csv"
    feature_importance_plot_path = out / "feature_importance.png"
    metrics_history_path = out / "metrics_history.csv"
    metrics_history_plot_path = out / "metrics_history.png"
    time_series_cv_metrics_path = out / "time_series_cv_metrics.csv"
    time_series_cv_plot_path = out / "time_series_cv_metrics.png"
    roc_curve_path = out / "roc_auc_ovr.png"

    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model_state_dict": model.cpu().state_dict(),
                "label_encoder": label_encoder,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "params": params,
                "weather_condition_labels": WEATHER_CONDITION_LABELS,
            },
            file,
        )
    metrics["metrics_history_csv"] = str(metrics_history_path)
    metrics["feature_importance_csv"] = str(feature_importance_path)
    metrics["time_series_cv_metrics_csv"] = str(time_series_cv_metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history.to_csv(metrics_history_path, index=False)
    pd.DataFrame({"feature": feature_columns, "importance": np.nan}).to_csv(
        feature_importance_path, index=False
    )
    pd.DataFrame().to_csv(time_series_cv_metrics_path, index=False)
    time_series_cv_plot_path.write_bytes(b"")
    feature_importance_plot_path.write_bytes(b"")
    pd.DataFrame(
        {
            "observed_at": pd.Series(test_times).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_weather_condition": y_test_raw,
            "predicted_weather_condition": predicted_conditions,
            "prediction_confidence": probabilities.max(axis=1),
        }
    ).to_csv(predictions_path, index=False)
    _write_confusion_matrix(confusion_matrix_path, y_test, predicted_labels, label_encoder.classes_)
    _write_metrics_history_plot(metrics_history_plot_path, history)
    roc_curve_written = _write_roc_curve(
        roc_curve_path, y_test, probabilities, label_encoder.classes_
    )

    if log_to_mlflow:
        _log_lstm_mlflow_run(
            experiment_name=experiment_name,
            metrics=metrics,
            output_dir=out,
            label_encoder=label_encoder,
            training_params=params,
            metrics_history=history,
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


def _build_lstm_sequences(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    values = data[feature_columns].to_numpy(dtype=np.float32)
    labels = data[target_column].astype(str).to_numpy()
    times = data["observed_at"].reset_index(drop=True)
    if len(data) <= sequence_length:
        raise ValueError("Not enough rows to build LSTM sequences.")
    sequences = np.stack(
        [values[index - sequence_length : index] for index in range(sequence_length, len(data))]
    )
    return sequences, labels[sequence_length:], times.iloc[sequence_length:].reset_index(drop=True)


def _fit_lstm(
    *,
    model: WeatherLSTM,
    train_loader: DataLoader,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
        for split_name, x_values, y_values in [
            ("train", x_train, y_train),
            ("test", x_test, y_test),
        ]:
            probabilities = _predict_lstm_probabilities(model, x_values, device)
            predicted = probabilities.argmax(axis=1)
            rows.append(
                {
                    "iteration": epoch,
                    "split": split_name,
                    "accuracy": float((predicted == y_values).mean()),
                    "precision_weighted": float(
                        precision_score(y_values, predicted, average="weighted", zero_division=0)
                    ),
                    "recall_weighted": float(
                        recall_score(y_values, predicted, average="weighted", zero_division=0)
                    ),
                    "f1_macro": float(
                        f1_score(y_values, predicted, average="macro", zero_division=0)
                    ),
                    "f1_weighted": float(
                        f1_score(y_values, predicted, average="weighted", zero_division=0)
                    ),
                    "roc_auc_ovr_weighted": np.nan,
                    "log_loss": float(
                        log_loss(
                            y_values,
                            probabilities,
                            labels=list(range(probabilities.shape[1])),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _predict_lstm_probabilities(
    model: WeatherLSTM, values: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(values, dtype=torch.float32).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def _log_lstm_mlflow_run(
    *,
    experiment_name: str,
    metrics: dict[str, object],
    output_dir: Path,
    label_encoder: LabelEncoder,
    training_params: LSTMTrainingParams,
    metrics_history: pd.DataFrame,
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="lstm-historical-forecast-weather-condition") as run:
        mlflow.log_params(
            {
                "model_type": "LSTM",
                "target_column": metrics["target_column"],
                "classes": ",".join(label_encoder.classes_),
                "training_scope": metrics["training_scope"],
                "sequence_length": training_params.sequence_length,
                "hidden_size": training_params.hidden_size,
                "num_layers": training_params.num_layers,
                "dropout": training_params.dropout,
                "epochs": training_params.epochs,
                "batch_size": training_params.batch_size,
                "learning_rate": training_params.learning_rate,
            }
        )
        mlflow.set_tags(
            {
                "model_status": "accepted" if metrics["accepted"] else "rejected",
                "acceptance_metric": metrics["acceptance_metric"],
                "acceptance_threshold": str(metrics["acceptance_threshold"]),
            }
        )
        mlflow.log_metrics(
            {key: float(metrics[key]) for key in SUMMARY_METRIC_NAMES if metrics[key] is not None}
        )
        _log_mlflow_metric_batches(run.info.run_id, [])
        mlflow.log_artifacts(str(output_dir))
