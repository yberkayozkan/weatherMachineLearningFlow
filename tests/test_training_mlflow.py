from types import SimpleNamespace

import numpy as np
import pandas as pd

from weather_ml import training


def _training_frame(row_count: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            column: np.linspace(0.0, 1.0, row_count)
            for column in training.DEFAULT_FEATURE_COLUMNS
        }
    )
    frame["observed_at"] = pd.date_range("2026-01-01", periods=row_count, freq="h")
    frame[training.DEFAULT_TARGET] = [
        ["clear", "rain", "cloudy"][index % 3] for index in range(row_count)
    ]
    frame["is_weekend"] = False
    frame["weather_condition_lag_4h_code"] = 0
    frame["weather_condition_lag_12h_code"] = 0
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
                "f1_weighted": 0.75,
                "roc_auc_ovr_weighted": 0.85,
                "log_loss": 0.2,
            }
            for index in range(row_count)
        ]
    )


def test_build_mlflow_history_metrics_preserves_names_steps_and_skips_missing_values() -> None:
    history = _metrics_history(2)
    history.loc[0, "roc_auc_ovr_weighted"] = np.nan
    history.loc[1, "log_loss"] = None

    metrics = training._build_mlflow_history_metrics(history)

    assert len(metrics) == 10
    assert {(metric.key, metric.step) for metric in metrics} >= {
        ("train_accuracy", 1),
        ("test_f1_weighted", 2),
    }
    assert not any(metric.key == "train_roc_auc_ovr_weighted" for metric in metrics)
    assert not any(metric.key == "test_log_loss" for metric in metrics)


def test_log_mlflow_metric_batches_splits_1200_metrics(monkeypatch) -> None:
    captured_batches: list[list[object]] = []

    class FakeClient:
        def log_batch(self, run_id, metrics=()):
            assert run_id == "run-123"
            captured_batches.append(list(metrics))

    monkeypatch.setattr(training, "MlflowClient", FakeClient)
    metrics = training._build_mlflow_history_metrics(_metrics_history(200))

    training._log_mlflow_metric_batches("run-123", metrics)

    assert len(metrics) == 1200
    assert [len(batch) for batch in captured_batches] == [500, 500, 200]


def test_log_mlflow_run_uploads_summary_as_batch_and_artifacts_once(monkeypatch, tmp_path) -> None:
    calls: dict[str, list[object]] = {
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
    monkeypatch.setattr(training.mlflow, "log_params", lambda values: None)
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
        "acceptance_metric": "f1_weighted",
        "acceptance_threshold": 0.4,
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
    )

    assert len(calls["summary"]) == 1
    assert calls["summary"][0] == {name: 0.5 for name in training.SUMMARY_METRIC_NAMES}
    assert len(calls["history"]) == 1
    assert calls["history"][0][0] == "run-123"
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
    assert artifacts.roc_curve_path is not None
    assert artifacts.roc_curve_path.exists()

    feature_importance = pd.read_csv(artifacts.feature_importance_path)
    assert list(feature_importance.columns) == ["feature", "importance"]
    assert set(feature_importance["feature"]) == set(training.DEFAULT_FEATURE_COLUMNS)
    assert feature_importance["importance"].is_monotonic_decreasing
