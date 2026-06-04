import numpy as np
import pandas as pd

from weather_ml import drift


def test_population_stability_index_is_near_zero_for_stable_distribution() -> None:
    value = drift.population_stability_index([50, 50], [50, 50])

    assert value == 0


def test_population_stability_index_increases_for_shifted_distribution() -> None:
    mild = drift.population_stability_index([50, 50], [40, 60])
    heavy = drift.population_stability_index([50, 50], [5, 95])

    assert mild > 0
    assert heavy > mild


def test_build_drift_report_handles_missing_baseline() -> None:
    report = drift.build_drift_report(
        baseline=None,
        current_frame=pd.DataFrame({"feature": [1.0, 2.0]}),
        feature_columns=["feature"],
    )

    assert report["status"] == "no_baseline"
    assert report["summary"]["current_row_count"] == 2


def test_build_drift_report_handles_missing_current_rows() -> None:
    baseline = drift.build_feature_distribution_baseline(
        pd.DataFrame({"feature": np.arange(10)}),
        ["feature"],
    )

    report = drift.build_drift_report(
        baseline=baseline,
        current_frame=pd.DataFrame({"feature": []}),
        feature_columns=["feature"],
    )

    assert report["status"] == "no_current_rows"


def test_build_drift_report_flags_drifted_features() -> None:
    baseline = drift.build_feature_distribution_baseline(
        pd.DataFrame({"feature": np.arange(100)}),
        ["feature"],
    )

    report = drift.build_drift_report(
        baseline=baseline,
        current_frame=pd.DataFrame({"feature": np.arange(90, 100)}),
        feature_columns=["feature"],
        warning_threshold=0.10,
        drift_threshold=0.20,
    )

    assert report["status"] == "drifted"
    assert report["features"]["feature"]["psi"] >= 0.20
