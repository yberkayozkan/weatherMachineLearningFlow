import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weather_ml import training


def _training_frame(row_count: int = 240) -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: np.linspace(0.0, 1.0, row_count) for column in training.DEFAULT_FEATURE_COLUMNS}
    )
    frame["observed_at"] = pd.date_range("2026-01-01", periods=row_count, freq="h")
    frame[training.DEFAULT_TARGET] = [
        ["clear", "rain", "cloudy"][index % 3] for index in range(row_count)
    ]
    frame["is_weekend"] = False
    frame["weather_condition_lag_4h_code"] = 0
    frame["weather_condition_lag_12h_code"] = 0
    return frame


def _historical_forecast_training_frame(row_count: int = 360) -> pd.DataFrame:
    frame = _training_frame(row_count)
    frame["observed_at"] = pd.date_range("2021-03-20", periods=row_count, freq="h")
    frame["cape"] = np.linspace(0.0, 200.0, row_count)
    frame["freezing_level_height"] = np.linspace(250.0, 2800.0, row_count)
    frame["uv_index"] = np.linspace(0.0, 8.0, row_count)
    return frame


def _metrics_history(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": index + 1,
                "split": "train" if index % 2 == 0 else "test",
                "accuracy": 0.9,
                "precision_weighted": 0.8,
                "recall_weighted": 0.7,
                "f1_macro": 0.65,
                "f1_weighted": 0.75,
                "roc_auc_ovr_weighted": 0.85,
                "log_loss": 0.2,
            }
            for index in range(row_count)
        ]
    )


def _cv_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold,
                **{name: 0.4 + fold / 10 for name in training.SUMMARY_METRIC_NAMES},
            }
            for fold in range(1, 4)
        ]
    )


def test_temporal_split_applies_exact_gap_to_cv_and_final_holdout() -> None:
    data = training._prepare_training_frame(
        _training_frame(), target_column=training.DEFAULT_TARGET
    )
    split = training._build_temporal_validation_split(data, test_fraction=0.2, cv_splits=3)

    assert split.final_test.index[0] - split.final_train.index[-1] - 1 == 24
    for train_indices, validation_indices in split.cv_indices:
        assert validation_indices[0] - train_indices[-1] - 1 == 24
        assert train_indices[-1] < validation_indices[0]


def test_build_mlflow_history_metrics_preserves_names_steps_and_skips_missing_values() -> None:
    history = _metrics_history(2)
    history.loc[0, "roc_auc_ovr_weighted"] = np.nan
    history.loc[1, "log_loss"] = None

    metrics = training._build_mlflow_history_metrics(history)

    assert len(metrics) == 12
    assert {(metric.key, metric.step) for metric in metrics} >= {
        ("train_accuracy", 1),
        ("test_f1_macro", 2),
        ("test_f1_weighted", 2),
    }
    assert not any(metric.key == "train_roc_auc_ovr_weighted" for metric in metrics)
    assert not any(metric.key == "test_log_loss" for metric in metrics)


def test_sqrt_balanced_sample_weight_softens_minority_class_weight() -> None:
    weights = training._build_sqrt_balanced_sample_weight(np.array([0, 0, 0, 1]))
    balanced_weights = np.array([2 / 3, 2 / 3, 2 / 3, 2])

    assert weights[-1] > weights[0]
    np.testing.assert_allclose(weights, np.sqrt(balanced_weights))


def test_log_mlflow_metric_batches_splits_1200_metrics(monkeypatch) -> None:
    captured_batches: list[list[object]] = []

    class FakeClient:
        def log_batch(self, run_id, metrics=()):
            assert run_id == "run-123"
            captured_batches.append(list(metrics))

    monkeypatch.setattr(training, "MlflowClient", FakeClient)
    metrics = training._build_mlflow_history_metrics(_metrics_history(200))

    training._log_mlflow_metric_batches("run-123", metrics)

    assert len(metrics) == 1400
    assert [len(batch) for batch in captured_batches] == [500, 500, 400]


def test_log_mlflow_run_uploads_summary_as_batch_and_artifacts_once(monkeypatch, tmp_path) -> None:
    calls: dict[str, list[object]] = {
        "params": [],
        "summary": [],
        "artifact_dirs": [],
        "single_artifacts": [],
        "history": [],
    }

    class RunContext:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(training.mlflow, "set_experiment", lambda value: None)
    monkeypatch.setattr(training.mlflow, "start_run", lambda **kwargs: RunContext())
    monkeypatch.setattr(
        training.mlflow, "log_params", lambda values: calls["params"].append(values)
    )
    monkeypatch.setattr(training.mlflow, "set_tags", lambda values: None)
    monkeypatch.setattr(
        training.mlflow,
        "log_metrics",
        lambda values: calls["summary"].append(values),
    )
    monkeypatch.setattr(
        training.mlflow,
        "log_artifacts",
        lambda path: calls["artifact_dirs"].append(path),
    )
    monkeypatch.setattr(
        training.mlflow,
        "log_artifact",
        lambda path: calls["single_artifacts"].append(path),
    )
    monkeypatch.setattr(
        training,
        "_log_mlflow_metric_batches",
        lambda run_id, metrics: calls["history"].append((run_id, metrics)),
    )

    metrics = {
        "target_column": training.DEFAULT_TARGET,
        "accepted": True,
        "acceptance_metric": "f1_macro",
        "acceptance_threshold": 0.4,
        "cv_mean_metrics": {name: 0.25 for name in training.SUMMARY_METRIC_NAMES},
        **{name: 0.5 for name in training.SUMMARY_METRIC_NAMES},
    }
    model = SimpleNamespace(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    label_encoder = SimpleNamespace(classes_=["clear", "rain"])

    training._log_mlflow_run(
        experiment_name="weather-test",
        model=model,
        metrics=metrics,
        output_dir=tmp_path,
        label_encoder=label_encoder,
        training_params=training.XGBoostTrainingParams(),
        metrics_history=_metrics_history(2),
        cv_metrics=_cv_metrics(),
        cv_splits=3,
    )

    assert len(calls["summary"]) == 1
    assert calls["summary"][0] == {
        **{name: 0.5 for name in training.SUMMARY_METRIC_NAMES},
        **{f"cv_mean_{name}": 0.25 for name in training.SUMMARY_METRIC_NAMES},
    }
    assert calls["params"][0]["cv_splits"] == 3
    assert calls["params"][0]["validation_gap_hours"] == 24
    assert calls["params"][0]["class_weighting"] == "sqrt_balanced_sample_weight"
    assert len(calls["history"]) == 1
    assert calls["history"][0][0] == "run-123"
    assert any(
        metric.key == "cv_validation_f1_macro" and metric.step == 3
        for metric in calls["history"][0][1]
    )
    assert calls["single_artifacts"] == []
    assert calls["artifact_dirs"] == [str(tmp_path)]


def test_training_excludes_legacy_summary_plot_and_preserves_diagnostic_artifacts(tmp_path) -> None:
    legacy_summary = tmp_path / "metrics_summary.png"
    legacy_summary.write_bytes(b"legacy")

    artifacts = training.train_weather_condition_model(
        _training_frame(),
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=training.XGBoostTrainingParams(n_estimators=2),
    )

    assert not hasattr(artifacts, "metrics_plot_path")
    assert not legacy_summary.exists()
    assert artifacts.confusion_matrix_path.exists()
    assert artifacts.feature_importance_path.exists()
    assert artifacts.feature_importance_plot_path.exists()
    assert artifacts.metrics_history_path.exists()
    assert artifacts.metrics_history_plot_path.exists()
    assert artifacts.time_series_cv_metrics_path.exists()
    assert artifacts.time_series_cv_plot_path.exists()
    assert artifacts.roc_curve_path is not None
    assert artifacts.roc_curve_path.exists()

    feature_importance = pd.read_csv(artifacts.feature_importance_path)
    assert list(feature_importance.columns) == ["feature", "importance"]
    assert set(feature_importance["feature"]) == set(training.DEFAULT_FEATURE_COLUMNS)
    assert feature_importance["importance"].is_monotonic_decreasing
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["validation_strategy"] == "time_series_split"
    assert metrics["cv_splits"] == 3
    assert metrics["validation_gap_hours"] == 24
    assert metrics["acceptance_metric"] == "f1_macro"
    assert metrics["accepted"] == (
        metrics["f1_macro"] >= training.XGBoostTrainingParams(n_estimators=2).acceptance_threshold
    )
    assert "f1_weighted" in metrics["cv_mean_metrics"]
    assert len(pd.read_csv(artifacts.time_series_cv_metrics_path)) == 3


def test_xgboost_removes_stale_roc_when_final_holdout_has_one_class(tmp_path) -> None:
    stale_roc = tmp_path / "roc_auc_ovr.png"
    stale_roc.write_bytes(b"stale")
    frame = _training_frame()
    frame.loc[frame.index[-48:], training.DEFAULT_TARGET] = "clear"

    artifacts = training.train_weather_condition_model(
        frame,
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=training.XGBoostTrainingParams(n_estimators=2),
    )

    assert artifacts.roc_curve_path is None
    assert not stale_roc.exists()


def test_historical_forecast_xgboost_uses_post_2021_enrichment_profile(tmp_path) -> None:
    artifacts = training.train_historical_forecast_xgboost_model(
        _historical_forecast_training_frame(),
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=training.XGBoostTrainingParams(n_estimators=2),
    )

    assert artifacts.model_path.name == "weather_condition_xgboost_historical_forecast_model.pkl"
    feature_importance = pd.read_csv(artifacts.feature_importance_path)
    assert set(feature_importance["feature"]) == set(training.HISTORICAL_FORECAST_FEATURE_COLUMNS)
    assert {"cape", "freezing_level_height", "uv_index"} <= set(feature_importance["feature"])
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["feature_columns"] == training.HISTORICAL_FORECAST_FEATURE_COLUMNS
    assert metrics["train_start"] >= training.HISTORICAL_FORECAST_TRAINING_START


def test_historical_forecast_xgboost_logs_to_separate_mlflow_identity(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def capture_mlflow_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(training, "_log_mlflow_run", capture_mlflow_run)
    training.train_historical_forecast_xgboost_model(
        _historical_forecast_training_frame(),
        output_dir=str(tmp_path),
        log_to_mlflow=True,
        training_params=training.XGBoostTrainingParams(n_estimators=2),
    )

    assert captured["experiment_name"] == "weather-condition-xgboost-historical-forecast"
    assert captured["mlflow_run_name"] == "xgboost-historical-forecast-weather-condition"
    assert captured["training_scope"] == "historical_forecast_2021_03_23_plus"
    assert captured["feature_columns"] == training.HISTORICAL_FORECAST_FEATURE_COLUMNS


def test_historical_forecast_workflow_invokes_dedicated_training_command() -> None:
    workflow = Path(".github/workflows/train-xgboost-historical-forecast-model.yml").read_text(
        encoding="utf-8"
    )

    assert "train_historical_forecast_xgboost_from_supabase" in workflow
    assert "artifacts/xgboost-historical-forecast-training" in workflow
    assert "OpenMeteo Supabase Pipeline" in workflow
