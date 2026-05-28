import json

import numpy as np
import pandas as pd

from weather_ml import lstm_training, training


def _historical_forecast_frame(row_count: int = 180) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            column: np.linspace(0.0, 1.0, row_count)
            for column in training.HISTORICAL_FORECAST_FEATURE_COLUMNS
        }
    )
    frame["observed_at"] = pd.date_range("2021-03-23", periods=row_count, freq="h")
    frame[training.DEFAULT_TARGET] = [
        ["clear", "rain", "cloudy"][index % 3] for index in range(row_count)
    ]
    frame["is_weekend"] = False
    frame["weather_condition_lag_4h_code"] = 0
    frame["weather_condition_lag_12h_code"] = 0
    return frame


def test_lstm_training_uses_2021_historical_forecast_features(tmp_path) -> None:
    artifacts = lstm_training.train_lstm_weather_condition_model(
        _historical_forecast_frame(),
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=lstm_training.LSTMTrainingParams(
            sequence_length=6,
            hidden_size=8,
            epochs=1,
            batch_size=32,
        ),
    )

    assert artifacts.model_path.name == "weather_condition_lstm_historical_forecast_model.pkl"
    assert artifacts.metrics_path.exists()
    assert artifacts.predictions_path.exists()
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["training_scope"] == "historical_forecast_2021_03_23_plus"
    assert metrics["feature_columns"] == training.HISTORICAL_FORECAST_FEATURE_COLUMNS
    assert metrics["lstm_params"]["sequence_length"] == 6
    assert metrics["train_start"] >= training.HISTORICAL_FORECAST_TRAINING_START


def test_lstm_sequence_builder_targets_current_row_after_history_window() -> None:
    frame = _historical_forecast_frame(10)
    sequences, labels, times = lstm_training._build_lstm_sequences(
        data=training._prepare_training_frame(
            frame,
            target_column=training.DEFAULT_TARGET,
            feature_columns=training.HISTORICAL_FORECAST_FEATURE_COLUMNS,
        ),
        feature_columns=training.HISTORICAL_FORECAST_FEATURE_COLUMNS,
        target_column=training.DEFAULT_TARGET,
        sequence_length=4,
    )

    assert sequences.shape[0] == 6
    assert labels[0] == frame.iloc[4][training.DEFAULT_TARGET]
    assert times.iloc[0] == frame.iloc[4]["observed_at"]
