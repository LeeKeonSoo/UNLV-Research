# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.10"]
# ///
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
    load_provider_registry,
)
from quality_effect_engine import (
    EffectCalibrationBundle,
    EffectInterval,
    EvidenceBin,
    QualityEffectObservation,
    QualityEvaluationRequest,
    calibrate_effect_bins,
    evaluate_quality_effect,
)


class IntervalFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    point: float
    lower: float
    upper: float
    samples: int


class BinFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str
    bin_id: str
    bin_order: int
    development: IntervalFixture
    heldout: IntervalFixture


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    contract_fixture_only_not_empirical_evidence: bool
    effect_metric_id: str
    effect_metric_artifact_sha256: str
    common_baseline_artifact_sha256: str
    provider_training_source_groups: frozenset[str]
    development_source_groups: frozenset[str]
    heldout_source_groups: frozenset[str]
    bins: tuple[BinFixture, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Quality effect engine v2 contract audit.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "validation" / "fixtures" / "quality_effect_engine_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "quality_effect_engine_v2_contract_audit.json",
    )
    return parser.parse_args()


def _fixture_provider() -> ProviderManifest:
    return ProviderManifest(
        provider_id="fixture-quality-provider",
        role=ProviderRole.QUALITY,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.ACTIVE,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="fixture-bin-assignment-v1",
        output_semantics="frozen-evidence-bin-id",
        supported_routes=("general_prose", "code_artifact"),
        supported_languages=("fixture",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=CalibrationEvidence(
            artifact_path="validation/frozen_contracts/quality-effect.json",
            artifact_sha256="a" * 64,
            scope_id="fixture-development",
        ),
        validation=ValidationEvidence(
            artifact_path="validation/frozen_contracts/quality-effect-confirmatory.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _calibration(bundle: FixtureBundle, provider: ProviderManifest) -> EffectCalibrationBundle:
    bins = tuple(
        EvidenceBin(
            route=effect.route,
            bin_id=effect.bin_id,
            bin_order=effect.bin_order,
            development=EffectInterval(**effect.development.model_dump()),
            heldout=EffectInterval(**effect.heldout.model_dump()),
            artifact_sha256="c" * 64,
        )
        for effect in bundle.bins
    )
    return EffectCalibrationBundle(
        provider_id=provider.provider_id,
        provider_identity_sha256=provider.identity_sha256(),
        effect_metric_id=bundle.effect_metric_id,
        effect_metric_artifact_sha256=bundle.effect_metric_artifact_sha256,
        common_baseline_artifact_sha256=bundle.common_baseline_artifact_sha256,
        bins=bins,
        provider_training_source_groups=bundle.provider_training_source_groups,
        development_source_groups=bundle.development_source_groups,
        heldout_source_groups=bundle.heldout_source_groups,
        all_arms_share_common_baseline=True,
        common_baseline_disjoint_from_all_bins=True,
        external_results_hidden=True,
        provider_bias_stress_passed=True,
        route_holdout_stress_passed=True,
    )


def _decision(provider: ProviderManifest, bin_id: str, calibration: EffectCalibrationBundle) -> dict[str, str | bool | None]:
    report = calibrate_effect_bins(calibration)
    request = QualityEvaluationRequest(
        QualityEffectObservation(
            chunk_uid=f"fixture:{bin_id}",
            route_state="routed",
            route="general_prose",
            provider_id=provider.provider_id,
            provider_identity_sha256=provider.identity_sha256(),
            bin_id=bin_id,
            observation_artifact_sha256="d" * 64,
        )
    )
    decision = evaluate_quality_effect(request, report, provider)
    return {
        "bin_id": bin_id,
        "decision": decision.decision.value,
        "reason_code": decision.reason_code,
        "may_mutate_curated_membership": decision.may_mutate_curated_membership,
    }


def main() -> None:
    args = parse_args()
    fixtures = FixtureBundle.model_validate_json(args.fixtures.read_text(encoding="utf-8"))
    provider = _fixture_provider()
    calibration = _calibration(fixtures, provider)
    report = calibrate_effect_bins(calibration)
    current_registry = load_provider_registry(ROOT / "configs" / "model_provider_registry_v1.json")
    current_provider = next(item for item in current_registry.providers if item.role is ProviderRole.QUALITY)
    current_decision = _decision(current_provider, "general-low", _calibration(fixtures, current_provider))
    payload = {
        "schema_version": "quality-effect-engine-v2-contract-audit-v1",
        "fixture_schema_version": fixtures.schema_version,
        "contract_fixture_only_not_empirical_evidence": fixtures.contract_fixture_only_not_empirical_evidence,
        "calibration_report": asdict(report),
        "fixture_active_provider_decisions": [
            _decision(provider, bin_id, calibration)
            for bin_id in ("general-low", "general-mid", "general-high")
        ],
        "current_registered_quality_provider_lifecycle": current_provider.lifecycle.value,
        "current_registered_provider_decision": current_decision,
        "empirical_runtime_activation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[quality-effect-engine-v2-audit] fixture_passed={report.passed} "
        f"current_provider={current_provider.lifecycle.value} output={args.output}"
    )


if __name__ == "__main__":
    main()
