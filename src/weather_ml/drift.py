from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PSI_WARNING_THRESHOLD = 0.10
DEFAULT_PSI_DRIFT_THRESHOLD = 0.20
PSI_EPSILON = 1e-6


def build_feature_distribution_baseline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for column in feature_columns:
        values = _numeric_values(frame[column])
        if values.size == 0:
            features[column] = {
                "edges": [],
                "counts": [],
                "total": 0,
                "status": "empty",
            }
            continue
        edges = _build_histogram_edges(values, bins=bins)
        counts, _ = np.histogram(values, bins=edges)
        features[column] = {
            "edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
            "total": int(counts.sum()),
            "status": "ready",
        }
    return {
        "schema_version": 1,
        "bins": bins,
        "row_count": int(len(frame)),
        "feature_count": len(feature_columns),
        "features": features,
    }


def write_feature_distribution_baseline(
    output_path: Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    baseline = build_feature_distribution_baseline(frame, feature_columns, bins=bins)
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def load_feature_distribution_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_drift_report(
    *,
    baseline: dict[str, Any] | None,
    current_frame: pd.DataFrame,
    feature_columns: list[str],
    warning_threshold: float = DEFAULT_PSI_WARNING_THRESHOLD,
    drift_threshold: float = DEFAULT_PSI_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    if baseline is None:
        return {
            "schema_version": 1,
            "status": "no_baseline",
            "summary": {
                "max_psi": None,
                "drifted_feature_count": 0,
                "warning_feature_count": 0,
                "current_row_count": int(len(current_frame)),
            },
            "features": {},
            "thresholds": _threshold_payload(warning_threshold, drift_threshold),
        }
    if current_frame.empty:
        return {
            "schema_version": 1,
            "status": "no_current_rows",
            "summary": {
                "max_psi": None,
                "drifted_feature_count": 0,
                "warning_feature_count": 0,
                "current_row_count": 0,
            },
            "features": {},
            "thresholds": _threshold_payload(warning_threshold, drift_threshold),
        }

    rows: dict[str, Any] = {}
    for column in feature_columns:
        feature_baseline = baseline.get("features", {}).get(column)
        rows[column] = _build_feature_psi_row(
            column=column,
            feature_baseline=feature_baseline,
            current_values=_numeric_values(current_frame[column]),
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )

    valid_psi = [
        float(row["psi"])
        for row in rows.values()
        if row["psi"] is not None and row["status"] in {"stable", "warning", "drifted"}
    ]
    drifted_count = sum(1 for row in rows.values() if row["status"] == "drifted")
    warning_count = sum(1 for row in rows.values() if row["status"] == "warning")
    status = "drifted" if drifted_count else "warning" if warning_count else "stable"
    return {
        "schema_version": 1,
        "status": status,
        "summary": {
            "max_psi": None if not valid_psi else float(max(valid_psi)),
            "drifted_feature_count": drifted_count,
            "warning_feature_count": warning_count,
            "current_row_count": int(len(current_frame)),
        },
        "features": rows,
        "thresholds": _threshold_payload(warning_threshold, drift_threshold),
    }


def population_stability_index(
    expected_counts: list[int] | np.ndarray,
    actual_counts: list[int] | np.ndarray,
) -> float:
    expected = _safe_proportions(np.asarray(expected_counts, dtype=float))
    actual = _safe_proportions(np.asarray(actual_counts, dtype=float))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _build_feature_psi_row(
    *,
    column: str,
    feature_baseline: dict[str, Any] | None,
    current_values: np.ndarray,
    warning_threshold: float,
    drift_threshold: float,
) -> dict[str, Any]:
    if not feature_baseline or feature_baseline.get("status") != "ready":
        return _feature_error(column, "missing_baseline")
    edges = np.asarray(feature_baseline.get("edges", []), dtype=float)
    baseline_counts = np.asarray(feature_baseline.get("counts", []), dtype=int)
    if len(edges) < 2 or len(baseline_counts) != len(edges) - 1:
        return _feature_error(column, "invalid_baseline")
    if current_values.size == 0:
        return _feature_error(column, "no_current_values")

    current_counts, _ = np.histogram(current_values, bins=edges)
    psi = population_stability_index(baseline_counts, current_counts)
    if psi >= drift_threshold:
        status = "drifted"
    elif psi >= warning_threshold:
        status = "warning"
    else:
        status = "stable"
    return {
        "feature": column,
        "status": status,
        "psi": psi,
        "baseline_total": int(baseline_counts.sum()),
        "current_total": int(current_counts.sum()),
        "current_out_of_range": int(current_values.size - current_counts.sum()),
    }


def _feature_error(column: str, status: str) -> dict[str, Any]:
    return {
        "feature": column,
        "status": status,
        "psi": None,
        "baseline_total": None,
        "current_total": None,
        "current_out_of_range": None,
    }


def _build_histogram_edges(values: np.ndarray, *, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        center = float(values[0])
        width = 0.5 if center == 0 else max(abs(center) * 0.01, 0.5)
        edges = np.array([center - width, center + width], dtype=float)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _numeric_values(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def _safe_proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 1 / len(counts), dtype=float)
    proportions = counts / total
    return np.clip(proportions, PSI_EPSILON, None)


def _threshold_payload(warning_threshold: float, drift_threshold: float) -> dict[str, float]:
    return {
        "warning": float(warning_threshold),
        "drift": float(drift_threshold),
    }
