from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contrastive_quality_provider import (
    ContrastiveEvidenceBundle,
    ContrastiveProviderError,
    ContrastiveQualityObservation,
    ContrastiveQualityProvider,
    FrozenTokenizerCompatibilityManifest,
    Precision,
    Sha256,
    hash_json,
)


class ContrastiveSampleRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    fixture_id: str = Field(min_length=1)
    parent_record_id: str = Field(min_length=1)
    contrastive_domain: str = Field(min_length=1)
    contrastive_route: str = Field(min_length=1)
    contrastive_scenario: str = Field(min_length=1)
    contrastive_source_id: str = Field(min_length=1)
    metamorphic_relation: str = Field(min_length=1)


class ContrastiveAuditInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: ContrastiveQualityProvider
    evidence: ContrastiveEvidenceBundle
    sample_rows: tuple[ContrastiveSampleRow, ...]
    sample_artifact_sha256: Sha256
    required_routes: tuple[str, ...]
    minimum_source_groups_per_route: int = Field(ge=2)
    empirical_effect_bins_by_route: dict[str, int]
    common_baseline_artifact_sha256: Sha256 | None
    provider_training_disjointness_artifact_sha256: Sha256 | None
    tokenizer_compatibility_manifest: FrozenTokenizerCompatibilityManifest | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "ContrastiveAuditInputs":
        if not self.sample_rows or not self.required_routes:
            raise ContrastiveProviderError("contrastive_audit_inputs_empty")
        if len({row.fixture_id for row in self.sample_rows}) != len(self.sample_rows):
            raise ContrastiveProviderError("contrastive_audit_sample_ids_not_unique")
        evidence_ids = {row.record_uid for row in self.evidence.records}
        sample_ids = {row.fixture_id for row in self.sample_rows}
        if not evidence_ids <= sample_ids:
            raise ContrastiveProviderError("contrastive_audit_evidence_outside_sample")
        if self.sample_artifact_sha256 != self.evidence.input_artifact_sha256:
            raise ContrastiveProviderError("contrastive_audit_sample_artifact_mismatch")
        if set(self.empirical_effect_bins_by_route) != set(self.required_routes):
            raise ContrastiveProviderError("contrastive_audit_effect_bin_routes_mismatch")
        manifest = self.tokenizer_compatibility_manifest
        if manifest is not None and (
            manifest.target_model_id != self.provider.target.model_id
            or manifest.target_revision != self.provider.target.revision
            or manifest.reference_model_id != self.provider.reference.model_id
            or manifest.reference_revision != self.provider.reference.revision
            or manifest.tokenizer_id != self.provider.tokenizer.tokenizer_id
            or manifest.tokenizer_revision != self.provider.tokenizer.revision
        ):
            raise ContrastiveProviderError("contrastive_audit_tokenizer_compatibility_scope_mismatch")
        return self


class DistributionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    count: int = Field(ge=1)
    minimum: float
    median: float
    mean: float
    maximum: float


class ExactCopyConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checked_copy_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)


class RouteAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    route: str
    sample_record_count: int = Field(ge=0)
    scored_record_count: int = Field(ge=0)
    source_group_count: int = Field(ge=0)
    source_groups: tuple[str, ...]
    scenario_counts: dict[str, int]
    target_nll: DistributionSummary | None
    reference_nll: DistributionSummary | None
    excess_nll: DistributionSummary | None
    target_entropy: DistributionSummary | None
    reference_entropy: DistributionSummary | None


class EvidenceGroupReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    route: str
    scenario: str
    relation: str
    scored_record_count: int = Field(ge=1)
    target_nll: DistributionSummary
    reference_nll: DistributionSummary
    excess_nll: DistributionSummary
    target_entropy: DistributionSummary
    reference_entropy: DistributionSummary


class RelationDeltaReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    route: str
    scenario: str
    relation: str
    paired_record_count: int = Field(ge=1)
    target_nll_delta: DistributionSummary
    reference_nll_delta: DistributionSummary
    excess_nll_delta: DistributionSummary
    target_entropy_delta: DistributionSummary
    reference_entropy_delta: DistributionSummary


class ContrastiveQualityAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["contrastive-quality-audit-v1"]
    status: Literal["blocked", "ready_for_effect_bin_experiment"]
    provider_id: str
    provider_identity_sha256: Sha256
    evidence_bundle_sha256: Sha256
    sample_artifact_sha256: Sha256
    tokenizer_compatibility_artifact_sha256: Sha256 | None
    sample_record_count: int = Field(ge=1)
    scored_record_count: int = Field(ge=1)
    omitted_record_count: int = Field(ge=0)
    omitted_relation_counts: dict[str, int]
    truncated_record_count: int = Field(ge=0)
    exact_copy_consistency: ExactCopyConsistency
    route_reports: tuple[RouteAuditReport, ...]
    evidence_group_reports: tuple[EvidenceGroupReport, ...]
    relation_delta_reports: tuple[RelationDeltaReport, ...]
    blocker_codes: tuple[str, ...]
    scalar_quality_score_emitted: Literal[False]
    threshold_decision_emitted: Literal[False]
    benchmark_outcomes_read: Literal[False]
    utility_read: Literal[False]
    runtime_activation: Literal[False]
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_report_hash(self) -> "ContrastiveQualityAuditReport":
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != hash_json(payload):
            raise ContrastiveProviderError("contrastive_audit_report_hash_mismatch")
        return self


def _distribution(values: tuple[float, ...]) -> DistributionSummary | None:
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ContrastiveProviderError("contrastive_audit_nonfinite_metric")
    return DistributionSummary(
        count=len(values),
        minimum=min(values),
        median=statistics.median(values),
        mean=statistics.fmean(values),
        maximum=max(values),
    )


def _same_observation(
    parent: ContrastiveQualityObservation,
    copy: ContrastiveQualityObservation,
) -> bool:
    return (
        parent.token_ids_sha256 == copy.token_ids_sha256
        and parent.scored_token_count == copy.scored_token_count
        and parent.target_nll == copy.target_nll
        and parent.reference_nll == copy.reference_nll
        and parent.excess_nll == copy.excess_nll
        and parent.target_entropy == copy.target_entropy
        and parent.reference_entropy == copy.reference_entropy
    )


def _exact_copy_consistency(
    sample_rows: tuple[ContrastiveSampleRow, ...],
    evidence_by_id: dict[str, ContrastiveQualityObservation],
) -> ExactCopyConsistency:
    parents: dict[tuple[str, str], ContrastiveSampleRow] = {}
    for row in sample_rows:
        if row.metamorphic_relation == "parent-retained-v1":
            parents[(row.contrastive_scenario, row.parent_record_id)] = row
    checked = 0
    mismatches = 0
    for row in sample_rows:
        if not row.metamorphic_relation.startswith("exact-copy-"):
            continue
        checked += 1
        parent_row = parents.get((row.contrastive_scenario, row.parent_record_id))
        parent = evidence_by_id.get(parent_row.fixture_id) if parent_row is not None else None
        copy = evidence_by_id.get(row.fixture_id)
        if parent is None or copy is None:
            mismatches += 1
            continue
        if not _same_observation(parent, copy):
            mismatches += 1
    return ExactCopyConsistency(checked_copy_count=checked, mismatch_count=mismatches)


def _route_report(
    route: str,
    sample_rows: tuple[ContrastiveSampleRow, ...],
    observations: tuple[ContrastiveQualityObservation, ...],
) -> RouteAuditReport:
    route_rows = tuple(row for row in sample_rows if row.contrastive_route == route)
    route_observations = tuple(row for row in observations if row.route == route)
    sources = tuple(sorted({row.contrastive_source_id for row in route_rows}))
    scenarios = Counter(row.contrastive_scenario for row in route_rows)
    return RouteAuditReport(
        route=route,
        sample_record_count=len(route_rows),
        scored_record_count=len(route_observations),
        source_group_count=len(sources),
        source_groups=sources,
        scenario_counts=dict(sorted(scenarios.items())),
        target_nll=_distribution(tuple(row.target_nll for row in route_observations)),
        reference_nll=_distribution(tuple(row.reference_nll for row in route_observations)),
        excess_nll=_distribution(tuple(row.excess_nll for row in route_observations)),
        target_entropy=_distribution(tuple(row.target_entropy for row in route_observations)),
        reference_entropy=_distribution(tuple(row.reference_entropy for row in route_observations)),
    )


def _required_distribution(values: tuple[float, ...]) -> DistributionSummary:
    summary = _distribution(values)
    if summary is None:
        raise ContrastiveProviderError("contrastive_audit_empty_distribution")
    return summary


def _evidence_group_reports(
    sample_rows: tuple[ContrastiveSampleRow, ...],
    evidence_by_id: dict[str, ContrastiveQualityObservation],
) -> tuple[EvidenceGroupReport, ...]:
    groups: dict[tuple[str, str, str], list[ContrastiveQualityObservation]] = defaultdict(list)
    for row in sample_rows:
        observation = evidence_by_id.get(row.fixture_id)
        if observation is not None:
            groups[(row.contrastive_route, row.contrastive_scenario, row.metamorphic_relation)].append(
                observation
            )
    return tuple(
        EvidenceGroupReport(
            route=key[0],
            scenario=key[1],
            relation=key[2],
            scored_record_count=len(values),
            target_nll=_required_distribution(tuple(item.target_nll for item in values)),
            reference_nll=_required_distribution(tuple(item.reference_nll for item in values)),
            excess_nll=_required_distribution(tuple(item.excess_nll for item in values)),
            target_entropy=_required_distribution(tuple(item.target_entropy for item in values)),
            reference_entropy=_required_distribution(tuple(item.reference_entropy for item in values)),
        )
        for key, values in sorted(groups.items())
    )


def _relation_delta_reports(
    sample_rows: tuple[ContrastiveSampleRow, ...],
    evidence_by_id: dict[str, ContrastiveQualityObservation],
) -> tuple[RelationDeltaReport, ...]:
    parent_rows = {
        (row.contrastive_route, row.contrastive_scenario, row.parent_record_id): row
        for row in sample_rows
        if row.metamorphic_relation == "parent-retained-v1"
    }
    deltas: dict[tuple[str, str, str], list[tuple[float, float, float, float, float]]] = defaultdict(list)
    for row in sample_rows:
        if row.metamorphic_relation == "parent-retained-v1":
            continue
        parent_row = parent_rows.get(
            (row.contrastive_route, row.contrastive_scenario, row.parent_record_id)
        )
        parent = evidence_by_id.get(parent_row.fixture_id) if parent_row is not None else None
        child = evidence_by_id.get(row.fixture_id)
        if parent is None or child is None:
            continue
        deltas[(row.contrastive_route, row.contrastive_scenario, row.metamorphic_relation)].append(
            (
                child.target_nll - parent.target_nll,
                child.reference_nll - parent.reference_nll,
                child.excess_nll - parent.excess_nll,
                child.target_entropy - parent.target_entropy,
                child.reference_entropy - parent.reference_entropy,
            )
        )
    return tuple(
        RelationDeltaReport(
            route=key[0],
            scenario=key[1],
            relation=key[2],
            paired_record_count=len(values),
            target_nll_delta=_required_distribution(tuple(item[0] for item in values)),
            reference_nll_delta=_required_distribution(tuple(item[1] for item in values)),
            excess_nll_delta=_required_distribution(tuple(item[2] for item in values)),
            target_entropy_delta=_required_distribution(tuple(item[3] for item in values)),
            reference_entropy_delta=_required_distribution(tuple(item[4] for item in values)),
        )
        for key, values in sorted(deltas.items())
    )


def build_contrastive_quality_audit(inputs: ContrastiveAuditInputs) -> ContrastiveQualityAuditReport:
    evidence_by_id = {row.record_uid: row for row in inputs.evidence.records}
    omitted = tuple(row for row in inputs.sample_rows if row.fixture_id not in evidence_by_id)
    omitted_relations = Counter(row.metamorphic_relation for row in omitted)
    route_reports = tuple(
        _route_report(route, inputs.sample_rows, inputs.evidence.records)
        for route in inputs.required_routes
    )
    blockers: list[str] = []
    if inputs.provider.reference.precision in {Precision.INT8, Precision.INT4} and (
        inputs.provider.reference.quantization_validation_artifact_sha256 is None
    ):
        blockers.append("reference_quantization_unvalidated")
    if inputs.tokenizer_compatibility_manifest is None:
        blockers.append("shared_tokenizer_compatibility_unverified")
    if inputs.provider_training_disjointness_artifact_sha256 is None:
        blockers.append("provider_training_disjointness_unverified")
    if inputs.common_baseline_artifact_sha256 is None:
        blockers.append("common_baseline_missing")
    for route_report in route_reports:
        if route_report.source_group_count < inputs.minimum_source_groups_per_route:
            blockers.append(f"insufficient_source_groups:{route_report.route}")
        if inputs.empirical_effect_bins_by_route[route_report.route] < 3:
            blockers.append(f"empirical_effect_bins_missing:{route_report.route}")
    if any(relation != "empty-payload-v1" for relation in omitted_relations):
        blockers.append("unexpected_unscored_records")
    exact_copy_consistency = _exact_copy_consistency(inputs.sample_rows, evidence_by_id)
    if exact_copy_consistency.mismatch_count:
        blockers.append("exact_copy_score_inconsistency")
    payload = {
        "schema_version": "contrastive-quality-audit-v1",
        "status": "blocked" if blockers else "ready_for_effect_bin_experiment",
        "provider_id": inputs.provider.provider_id,
        "provider_identity_sha256": inputs.provider.identity_sha256(),
        "evidence_bundle_sha256": inputs.evidence.evidence_bundle_sha256,
        "sample_artifact_sha256": inputs.sample_artifact_sha256,
        "tokenizer_compatibility_artifact_sha256": (
            None
            if inputs.tokenizer_compatibility_manifest is None
            else inputs.tokenizer_compatibility_manifest.artifact_sha256
        ),
        "sample_record_count": len(inputs.sample_rows),
        "scored_record_count": len(inputs.evidence.records),
        "omitted_record_count": len(omitted),
        "omitted_relation_counts": dict(sorted(omitted_relations.items())),
        "truncated_record_count": sum(row.truncated for row in inputs.evidence.records),
        "exact_copy_consistency": exact_copy_consistency.model_dump(mode="json"),
        "route_reports": [report.model_dump(mode="json") for report in route_reports],
        "evidence_group_reports": [
            report.model_dump(mode="json")
            for report in _evidence_group_reports(inputs.sample_rows, evidence_by_id)
        ],
        "relation_delta_reports": [
            report.model_dump(mode="json")
            for report in _relation_delta_reports(inputs.sample_rows, evidence_by_id)
        ],
        "blocker_codes": tuple(blockers),
        "scalar_quality_score_emitted": False,
        "threshold_decision_emitted": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "runtime_activation": False,
    }
    return ContrastiveQualityAuditReport(report_sha256=hash_json(payload), **payload)


__all__ = [
    "ContrastiveAuditInputs",
    "ContrastiveQualityAuditReport",
    "build_contrastive_quality_audit",
]
