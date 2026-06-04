from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

PRODUCTION_ALIAS = "production"


@dataclass(frozen=True)
class ProductionModelSnapshot:
    model_name: str
    alias: str
    version: str
    run_id: str
    metrics: dict[str, float]
    baseline: dict[str, Any] | None


@dataclass(frozen=True)
class PromotionDecision:
    should_promote: bool
    reason: str
    candidate_version: str | None
    production_version: str | None
    candidate_metric: float | None
    production_metric: float | None
    min_improvement: float
    promoted_alias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_promote": self.should_promote,
            "reason": self.reason,
            "candidate_version": self.candidate_version,
            "production_version": self.production_version,
            "candidate_metric": self.candidate_metric,
            "production_metric": self.production_metric,
            "min_improvement": self.min_improvement,
            "promoted_alias": self.promoted_alias,
        }


def configure_mlflow_tracking_from_env() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def get_production_model_snapshot(
    *,
    model_name: str,
    artifact_dir: Path,
    alias: str = PRODUCTION_ALIAS,
    client: MlflowClient | None = None,
) -> ProductionModelSnapshot | None:
    client = client or MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        logger.info(
            "production alias is not available: model=%s alias=%s error=%s",
            model_name,
            alias,
            exc,
        )
        return None

    run = client.get_run(version.run_id)
    baseline = _download_json_artifact(
        client=client,
        run_id=version.run_id,
        artifact_path="feature_distribution_baseline.json",
        artifact_dir=artifact_dir,
    )
    return ProductionModelSnapshot(
        model_name=model_name,
        alias=alias,
        version=str(version.version),
        run_id=version.run_id,
        metrics={key: float(value) for key, value in run.data.metrics.items()},
        baseline=baseline,
    )


def get_latest_model_version(
    *,
    model_name: str,
    client: MlflowClient | None = None,
) -> str | None:
    client = client or MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        return None
    return str(max(int(version.version) for version in versions))


def build_promotion_decision(
    *,
    candidate_metrics: dict[str, Any],
    candidate_version: str | None,
    production_snapshot: ProductionModelSnapshot | None,
    promotion_metric: str,
    min_improvement: float,
    alias: str = PRODUCTION_ALIAS,
) -> PromotionDecision:
    candidate_metric = _metric_value(candidate_metrics, promotion_metric)
    candidate_accepted = bool(candidate_metrics.get("accepted", False))
    if candidate_version is None:
        return PromotionDecision(
            should_promote=False,
            reason="missing_candidate_version",
            candidate_version=None,
            production_version=None if production_snapshot is None else production_snapshot.version,
            candidate_metric=candidate_metric,
            production_metric=None,
            min_improvement=min_improvement,
            promoted_alias=alias,
        )
    if candidate_metric is None:
        return PromotionDecision(
            should_promote=False,
            reason="missing_candidate_metric",
            candidate_version=candidate_version,
            production_version=None if production_snapshot is None else production_snapshot.version,
            candidate_metric=None,
            production_metric=None,
            min_improvement=min_improvement,
            promoted_alias=alias,
        )
    if not candidate_accepted:
        return PromotionDecision(
            should_promote=False,
            reason="candidate_not_accepted",
            candidate_version=candidate_version,
            production_version=None if production_snapshot is None else production_snapshot.version,
            candidate_metric=candidate_metric,
            production_metric=None,
            min_improvement=min_improvement,
            promoted_alias=alias,
        )
    if production_snapshot is None:
        return PromotionDecision(
            should_promote=True,
            reason="bootstrap_no_production_alias",
            candidate_version=candidate_version,
            production_version=None,
            candidate_metric=candidate_metric,
            production_metric=None,
            min_improvement=min_improvement,
            promoted_alias=alias,
        )

    production_metric = production_snapshot.metrics.get(promotion_metric)
    if production_metric is None:
        return PromotionDecision(
            should_promote=False,
            reason="missing_production_metric",
            candidate_version=candidate_version,
            production_version=production_snapshot.version,
            candidate_metric=candidate_metric,
            production_metric=None,
            min_improvement=min_improvement,
            promoted_alias=alias,
        )
    should_promote = candidate_metric >= production_metric + min_improvement
    return PromotionDecision(
        should_promote=should_promote,
        reason="metric_improved" if should_promote else "insufficient_improvement",
        candidate_version=candidate_version,
        production_version=production_snapshot.version,
        candidate_metric=candidate_metric,
        production_metric=float(production_metric),
        min_improvement=min_improvement,
        promoted_alias=alias,
    )


def promote_model_version(
    *,
    model_name: str,
    version: str,
    alias: str = PRODUCTION_ALIAS,
    client: MlflowClient | None = None,
) -> None:
    client = client or MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)


def write_promotion_decision(output_path: Path, decision: PromotionDecision) -> None:
    output_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")


def _download_json_artifact(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    artifact_dir: Path,
) -> dict[str, Any] | None:
    try:
        local_path = client.download_artifacts(run_id, artifact_path, str(artifact_dir))
    except MlflowException as exc:
        logger.info(
            "could not download MLflow artifact: run_id=%s path=%s error=%s",
            run_id,
            artifact_path,
            exc,
        )
        return None
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if value is None:
        return None
    return float(value)
