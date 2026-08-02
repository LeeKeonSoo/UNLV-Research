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

from coverage_engine import (
    CoverageChunk,
    CoverageRequest,
    CoverageStratum,
    CoverageView,
    FrozenSimilarity,
    RepresentativeFamily,
    StratumState,
    evaluate_coverage,
)
from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
    load_provider_registry,
)


class ChunkFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uid: str
    token_count: int


class StratumFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stratum_id: str
    view: CoverageView
    member_uids: frozenset[str]
    state: StratumState
    evidence_artifact_sha256: str


class FamilyFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family_id: str
    member_uids: frozenset[str]
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
    proposed_survivors: frozenset[str]
    strata: tuple[StratumFixture, ...]
    redundancy_families: tuple[FamilyFixture, ...]
    similarities: tuple[SimilarityFixture, ...]
    similarity_artifact_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Coverage engine v2 contract audit.")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "validation" / "fixtures" / "coverage_engine_v2_cases.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "coverage_engine_v2_contract_audit.json",
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
        calibration=CalibrationEvidence(
            artifact_path="validation/frozen_contracts/coverage-calibration.json",
            artifact_sha256="a" * 64,
            scope_id="fixture-development",
        ),
        validation=ValidationEvidence(
            artifact_path="validation/frozen_contracts/coverage-confirmatory.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _request(fixtures: FixtureBundle, provider: ProviderManifest) -> CoverageRequest:
    return CoverageRequest(
        chunks=tuple(CoverageChunk(item.uid, item.token_count) for item in fixtures.chunks),
        proposed_survivors=fixtures.proposed_survivors,
        strata=tuple(
            CoverageStratum(item.stratum_id, item.view, item.member_uids, item.state, item.evidence_artifact_sha256)
            for item in fixtures.strata
        ),
        redundancy_families=tuple(
            RepresentativeFamily(item.family_id, item.member_uids, item.evidence_artifact_sha256)
            for item in fixtures.redundancy_families
        ),
        similarities=tuple(
            FrozenSimilarity(item.left_uid, item.right_uid, item.similarity, fixtures.similarity_artifact_sha256)
            for item in fixtures.similarities
        ),
        exclusions=(),
        provider_id=provider.provider_id,
        provider_identity_sha256=provider.identity_sha256(),
    )


def main() -> None:
    args = parse_args()
    fixtures = FixtureBundle.model_validate_json(args.fixtures.read_text(encoding="utf-8"))
    fixture_provider = _fixture_provider()
    fixture_decision = evaluate_coverage(_request(fixtures, fixture_provider), fixture_provider)
    registry = load_provider_registry(ROOT / "configs" / "model_provider_registry_v1.json")
    current_provider = next(item for item in registry.providers if item.role is ProviderRole.SEMANTIC)
    current_decision = evaluate_coverage(_request(fixtures, current_provider), current_provider)
    payload = {
        "schema_version": "coverage-engine-v2-contract-audit-v1",
        "fixture_schema_version": fixtures.schema_version,
        "contract_fixture_only_not_empirical_evidence": fixtures.contract_fixture_only_not_empirical_evidence,
        "fixture_active_provider_decision": asdict(fixture_decision),
        "current_registered_semantic_provider_lifecycle": current_provider.lifecycle.value,
        "current_registered_provider_decision": asdict(current_decision),
        "empirical_runtime_activation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[coverage-engine-v2-audit] fixture={fixture_decision.status.value} "
        f"current_provider={current_provider.lifecycle.value} output={args.output}"
    )


if __name__ == "__main__":
    main()
