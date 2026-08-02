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

from coverage_engine import CoverageChunk, CoverageStratum, CoverageView, FrozenSimilarity, RepresentativeFamily, StratumState
from joint_selector import (
    JointGateBundle,
    JointGateOrigin,
    JointProfileName,
    JointSelectionRequest,
    evaluate_joint_selection,
    load_current_joint_gates,
    load_joint_profiles,
)
from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
    load_provider_registry,
)
from quality_effect_calibration import EffectDirection
from quality_effect_engine import QualityEffectDecision, QualityEffectDecisionName


class ChunkFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uid: str
    token_count: int


class QualityFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_uid: str
    decision: QualityEffectDecisionName
    reason_code: str
    effect_direction: EffectDirection


class FamilyFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family_id: str
    member_uids: frozenset[str]
    evidence_artifact_sha256: str


class StratumFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stratum_id: str
    view: CoverageView
    member_uids: frozenset[str]
    state: StratumState
    evidence_artifact_sha256: str


class SimilarityFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    left_uid: str
    right_uid: str
    similarity: float


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    contract_fixture_only_not_empirical_evidence: bool
    chunks: tuple[ChunkFixture, ...]
    quality_decisions: tuple[QualityFixture, ...]
    quality_evidence_artifact_hashes: tuple[str, ...]
    redundancy_families: tuple[FamilyFixture, ...]
    coverage_strata: tuple[StratumFixture, ...]
    similarities: tuple[SimilarityFixture, ...]
    similarity_artifact_sha256: str
    joint_gate_artifact_hashes: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the joint selector v1 contract audit.")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "validation" / "fixtures" / "joint_selector_v1_cases.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "joint_selector_v1_contract_audit.json",
    )
    return parser.parse_args()


def _fixture_provider() -> ProviderManifest:
    return ProviderManifest(
        provider_id="fixture-semantic-provider",
        role=ProviderRole.SEMANTIC,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.ACTIVE,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="fixture-frozen-similarity-v1",
        output_semantics="frozen-pairwise-similarity-and-stable-strata",
        supported_routes=("fixture",),
        supported_languages=("fixture",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=CalibrationEvidence(artifact_path="fixture-calibration.json", artifact_sha256="a" * 64, scope_id="fixture-development"),
        validation=ValidationEvidence(
            artifact_path="fixture-validation.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _request(fixtures: FixtureBundle, provider: ProviderManifest) -> JointSelectionRequest:
    return JointSelectionRequest(
        chunks=tuple(CoverageChunk(item.uid, item.token_count) for item in fixtures.chunks),
        redundancy_families=tuple(
            RepresentativeFamily(item.family_id, item.member_uids, item.evidence_artifact_sha256)
            for item in fixtures.redundancy_families
        ),
        quality_decisions=tuple(
            QualityEffectDecision(
                item.decision,
                item.reason_code,
                item.chunk_uid,
                item.effect_direction,
                fixtures.quality_evidence_artifact_hashes,
            )
            for item in fixtures.quality_decisions
        ),
        coverage_strata=tuple(
            CoverageStratum(item.stratum_id, item.view, item.member_uids, item.state, item.evidence_artifact_sha256)
            for item in fixtures.coverage_strata
        ),
        similarities=tuple(
            FrozenSimilarity(item.left_uid, item.right_uid, item.similarity, fixtures.similarity_artifact_sha256)
            for item in fixtures.similarities
        ),
        quality_provider_identity_sha256=fixtures.quality_evidence_artifact_hashes[0],
        semantic_provider_id=provider.provider_id,
        semantic_provider_identity_sha256=provider.identity_sha256(),
    )


def main() -> None:
    args = parse_args()
    fixtures = FixtureBundle.model_validate_json(args.fixtures.read_text(encoding="utf-8"))
    profiles = load_joint_profiles(ROOT / "configs" / "joint_selector_profiles_v1.json")
    fixture_provider = _fixture_provider()
    request = _request(fixtures, fixture_provider)
    ready_gates = JointGateBundle(JointGateOrigin.CONTRACT_FIXTURE, True, True, True, False, True, fixtures.joint_gate_artifact_hashes)
    normal_profile = profiles.by_name(JointProfileName.NORMAL)
    normal = evaluate_joint_selection(request, normal_profile, ready_gates, fixture_provider)
    replay = evaluate_joint_selection(request, normal_profile, ready_gates, fixture_provider)
    hard = evaluate_joint_selection(request, profiles.by_name(JointProfileName.HARD), ready_gates, fixture_provider)
    registry = load_provider_registry(ROOT / "configs" / "model_provider_registry_v1.json")
    current_provider = next(item for item in registry.providers if item.role is ProviderRole.SEMANTIC)
    blocked_gates = load_current_joint_gates(ROOT)
    current = evaluate_joint_selection(_request(fixtures, current_provider), normal_profile, blocked_gates, current_provider)
    payload = {
        "schema_version": "joint-selector-v1-contract-audit-v1",
        "fixture_schema_version": fixtures.schema_version,
        "contract_fixture_only_not_empirical_evidence": fixtures.contract_fixture_only_not_empirical_evidence,
        "fixture_normal_result": asdict(normal),
        "deterministic_replay_passed": normal == replay,
        "fixture_hard_result": asdict(hard),
        "current_semantic_provider_lifecycle": current_provider.lifecycle.value,
        "current_registered_evidence_result": asdict(current),
        "empirical_runtime_activation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[joint-selector-v1-audit] normal={normal.status.value} replay={normal == replay} "
        f"current={current.status.value} output={args.output}"
    )


if __name__ == "__main__":
    main()
