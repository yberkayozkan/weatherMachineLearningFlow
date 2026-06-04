from pathlib import Path
from types import SimpleNamespace

from weather_ml import model_registry


def test_promotion_bootstraps_when_no_production_and_candidate_is_accepted() -> None:
    decision = model_registry.build_promotion_decision(
        candidate_metrics={"accepted": True, "f1_macro": 0.45},
        candidate_version="7",
        production_snapshot=None,
        promotion_metric="f1_macro",
        min_improvement=0.02,
    )

    assert decision.should_promote
    assert decision.reason == "bootstrap_no_production_alias"


def test_promotion_happens_when_candidate_beats_production_threshold() -> None:
    production = model_registry.ProductionModelSnapshot(
        model_name="weather",
        alias="production",
        version="3",
        run_id="run-3",
        metrics={"f1_macro": 0.50},
        baseline=None,
    )

    decision = model_registry.build_promotion_decision(
        candidate_metrics={"accepted": True, "f1_macro": 0.52},
        candidate_version="4",
        production_snapshot=production,
        promotion_metric="f1_macro",
        min_improvement=0.02,
    )

    assert decision.should_promote
    assert decision.reason == "metric_improved"


def test_promotion_does_not_happen_when_improvement_is_too_small() -> None:
    production = model_registry.ProductionModelSnapshot(
        model_name="weather",
        alias="production",
        version="3",
        run_id="run-3",
        metrics={"f1_macro": 0.50},
        baseline=None,
    )

    decision = model_registry.build_promotion_decision(
        candidate_metrics={"accepted": True, "f1_macro": 0.519},
        candidate_version="4",
        production_snapshot=production,
        promotion_metric="f1_macro",
        min_improvement=0.02,
    )

    assert not decision.should_promote
    assert decision.reason == "insufficient_improvement"


def test_promotion_does_not_happen_for_rejected_candidate() -> None:
    decision = model_registry.build_promotion_decision(
        candidate_metrics={"accepted": False, "f1_macro": 0.90},
        candidate_version="4",
        production_snapshot=None,
        promotion_metric="f1_macro",
        min_improvement=0.02,
    )

    assert not decision.should_promote
    assert decision.reason == "candidate_not_accepted"


def test_promote_model_version_sets_alias_with_mlflow_client() -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            calls.append((name, alias, version))

    model_registry.promote_model_version(
        model_name="weather",
        version="5",
        alias="production",
        client=FakeClient(),
    )

    assert calls == [("weather", "production", "5")]


def test_get_latest_model_version_returns_highest_version() -> None:
    class FakeClient:
        def search_model_versions(self, filter_string: str):
            assert filter_string == "name = 'weather'"
            return [SimpleNamespace(version="1"), SimpleNamespace(version="12")]

    assert (
        model_registry.get_latest_model_version(model_name="weather", client=FakeClient())
        == "12"
    )


def test_write_promotion_decision_writes_json(tmp_path: Path) -> None:
    decision = model_registry.PromotionDecision(
        should_promote=False,
        reason="insufficient_improvement",
        candidate_version="2",
        production_version="1",
        candidate_metric=0.41,
        production_metric=0.40,
        min_improvement=0.02,
        promoted_alias="production",
    )

    output_path = tmp_path / "promotion_decision.json"
    model_registry.write_promotion_decision(output_path, decision)

    assert "insufficient_improvement" in output_path.read_text(encoding="utf-8")
