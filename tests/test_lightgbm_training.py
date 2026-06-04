import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from weather_ml import lightgbm_training, training


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


def _metrics_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": 1,
                "split": "train",
                "accuracy": 0.9,
                "precision_weighted": 0.8,
                "recall_weighted": 0.7,
                "f1_macro": 0.65,
                "f1_weighted": 0.75,
                "roc_auc_ovr_weighted": 0.85,
                "log_loss": 0.2,
            }
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


def test_lightgbm_training_writes_diagnostic_artifacts(tmp_path) -> None:
    artifacts = lightgbm_training.train_lightgbm_weather_condition_model(
        _training_frame(),
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=lightgbm_training.LightGBMTrainingParams(n_estimators=3),
    )

    assert artifacts.model_path.name == "weather_condition_lightgbm_model.pkl"
    assert artifacts.model_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.predictions_path.exists()
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
    assert set(feature_importance["feature"]) == set(training.DEFAULT_FEATURE_COLUMNS)
    assert feature_importance["importance"].is_monotonic_decreasing
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert "lightgbm_params" in metrics
    assert "xgboost_params" not in metrics
    assert metrics["validation_strategy"] == "time_series_split"
    assert metrics["cv_splits"] == 3
    assert metrics["validation_gap_hours"] == 24
    assert metrics["acceptance_metric"] == "f1_macro"
    assert len(pd.read_csv(artifacts.time_series_cv_metrics_path)) == 3


def test_lightgbm_history_uses_iteration_steps() -> None:
    calls: list[int] = []

    class FakeModel:
        def predict_proba(self, values, *, num_iteration):
            calls.append(num_iteration)
            return np.tile([[0.7, 0.2, 0.1]], (len(values), 1))

    values = pd.DataFrame({"value": [1, 2, 3]})
    history = lightgbm_training._build_lightgbm_metrics_history(
        model=FakeModel(),
        x_train=values,
        y_train=np.array([0, 1, 2]),
        x_test=values,
        y_test=np.array([0, 1, 2]),
        classes=np.array(["clear", "rain", "cloudy"]),
        total_iterations=3,
    )

    assert calls == [1, 1, 2, 2, 3, 3]
    assert history["iteration"].tolist() == [1, 1, 2, 2, 3, 3]


def test_lightgbm_mlflow_logs_params_metrics_history_and_artifacts(monkeypatch, tmp_path) -> None:
    calls: dict[str, list[object]] = {
        "experiments": [],
        "run_names": [],
        "params": [],
        "tags": [],
        "summary": [],
        "history": [],
        "artifacts": [],
    }

    class RunContext:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="lightgbm-run"))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        lightgbm_training.mlflow,
        "set_experiment",
        lambda value: calls["experiments"].append(value),
    )
    monkeypatch.setattr(
        lightgbm_training.mlflow,
        "start_run",
        lambda **kwargs: calls["run_names"].append(kwargs["run_name"]) or RunContext(),
    )
    monkeypatch.setattr(
        lightgbm_training.mlflow, "log_params", lambda value: calls["params"].append(value)
    )
    monkeypatch.setattr(
        lightgbm_training.mlflow, "set_tags", lambda value: calls["tags"].append(value)
    )
    monkeypatch.setattr(
        lightgbm_training.mlflow, "log_metrics", lambda value: calls["summary"].append(value)
    )
    monkeypatch.setattr(
        lightgbm_training.mlflow, "log_artifacts", lambda value: calls["artifacts"].append(value)
    )
    monkeypatch.setattr(
        lightgbm_training,
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
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
    )
    lightgbm_training._log_lightgbm_mlflow_run(
        experiment_name="weather-condition-lightgbm",
        model=model,
        metrics=metrics,
        output_dir=tmp_path,
        label_encoder=SimpleNamespace(classes_=["clear", "rain"]),
        training_params=lightgbm_training.LightGBMTrainingParams(),
        metrics_history=_metrics_history(),
        cv_metrics=_cv_metrics(),
        cv_splits=3,
    )

    assert calls["experiments"] == ["weather-condition-lightgbm"]
    assert calls["run_names"] == ["lightgbm-weather-condition"]
    assert calls["params"][0]["model_type"] == "LGBMClassifier"
    assert calls["params"][0]["cv_splits"] == 3
    assert calls["params"][0]["validation_gap_hours"] == 24
    assert calls["params"][0]["class_weighting"] == "sqrt_balanced_sample_weight"
    assert calls["summary"][0] == {
        **{name: 0.5 for name in training.SUMMARY_METRIC_NAMES},
        **{f"cv_mean_{name}": 0.25 for name in training.SUMMARY_METRIC_NAMES},
    }
    assert calls["history"][0][0] == "lightgbm-run"
    assert any(
        metric.key == "cv_validation_f1_macro" and metric.step == 3
        for metric in calls["history"][0][1]
    )
    assert calls["artifacts"] == [str(tmp_path)]


def test_lightgbm_mlflow_registers_pyfunc_model_when_model_path_is_available(
    monkeypatch, tmp_path
) -> None:
    calls: dict[str, list[object]] = {"registry": [], "history": []}

    class RunContext:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="lightgbm-run"))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(lightgbm_training.mlflow, "set_experiment", lambda value: None)
    monkeypatch.setattr(lightgbm_training.mlflow, "start_run", lambda **kwargs: RunContext())
    monkeypatch.setattr(lightgbm_training.mlflow, "log_params", lambda value: None)
    monkeypatch.setattr(lightgbm_training.mlflow, "set_tags", lambda value: None)
    monkeypatch.setattr(lightgbm_training.mlflow, "log_metrics", lambda value: None)
    monkeypatch.setattr(lightgbm_training.mlflow, "log_artifacts", lambda value: None)
    monkeypatch.setattr(
        lightgbm_training,
        "_log_mlflow_metric_batches",
        lambda run_id, metrics: calls["history"].append((run_id, metrics)),
    )
    monkeypatch.setattr(
        lightgbm_training,
        "_log_registered_pyfunc_model",
        lambda **kwargs: calls["registry"].append(kwargs),
    )

    model_path = tmp_path / "weather_condition_lightgbm_model.pkl"
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
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
    )

    lightgbm_training._log_lightgbm_mlflow_run(
        experiment_name="weather-condition-lightgbm",
        model=model,
        metrics=metrics,
        output_dir=tmp_path,
        label_encoder=SimpleNamespace(classes_=["clear", "rain"]),
        training_params=lightgbm_training.LightGBMTrainingParams(),
        metrics_history=_metrics_history(),
        cv_metrics=_cv_metrics(),
        cv_splits=3,
        model_path=model_path,
        registered_model_name="weather-lightgbm-prod",
    )

    assert calls["registry"] == [
        {"model_path": model_path, "registered_model_name": "weather-lightgbm-prod"}
    ]


def test_lightgbm_removes_stale_roc_when_final_holdout_has_one_class(tmp_path) -> None:
    stale_roc = tmp_path / "roc_auc_ovr.png"
    stale_roc.write_bytes(b"stale")
    frame = _training_frame()
    frame.loc[frame.index[-48:], training.DEFAULT_TARGET] = "clear"

    artifacts = lightgbm_training.train_lightgbm_weather_condition_model(
        frame,
        output_dir=str(tmp_path),
        log_to_mlflow=False,
        training_params=lightgbm_training.LightGBMTrainingParams(n_estimators=3),
    )

    assert artifacts.roc_curve_path is None
    assert not stale_roc.exists()
