from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

from development_corpus_inventory_contract import (
    DevelopmentCorpusInventoryManifest,
    InventoryStatus,
    SliceStatus,
)
from development_redundancy_gate_contract import (
    DevelopmentFixtureRecord,
    DevelopmentRedundancyGateError,
    DevelopmentRedundancyGateRegistry,
    DevelopmentRedundancyGateReport,
    RedundancyGateStatus,
    RedundancySliceInput,
    RelationCount,
    SliceRedundancyEvidence,
    hash_json,
)
from redundancy_v2 import RedundancyUnit, RelationType, classify_relation, formatting_canonical


@dataclass(frozen=True, slots=True)
class _SafeFamily:
    members: tuple[DevelopmentFixtureRecord, ...]
    relations: tuple[RelationType, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(slice_input: RedundancySliceInput) -> tuple[DevelopmentFixtureRecord, ...]:
    if not slice_input.path.is_file() or _sha256_file(slice_input.path) != slice_input.expected_sha256:
        raise DevelopmentRedundancyGateError(f"redundancy_slice_hash_mismatch:{slice_input.slice_id}")
    records: list[DevelopmentFixtureRecord] = []
    with slice_input.path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(DevelopmentFixtureRecord.model_validate_json(line))
    if not records or any(record.slice_id != slice_input.slice_id for record in records):
        raise DevelopmentRedundancyGateError(f"redundancy_slice_identity_mismatch:{slice_input.slice_id}")
    if len({record.fixture_id for record in records}) != len(records):
        raise DevelopmentRedundancyGateError(f"redundancy_fixture_ids_not_unique:{slice_input.slice_id}")
    return tuple(records)


def _safe_families(
    records: tuple[DevelopmentFixtureRecord, ...],
    registry: DevelopmentRedundancyGateRegistry,
) -> tuple[tuple[_SafeFamily, ...], Counter[RelationType]]:
    buckets: dict[str, list[DevelopmentFixtureRecord]] = defaultdict(list)
    for record in records:
        buckets[formatting_canonical(record.text)].append(record)
    families: list[_SafeFamily] = []
    counts: Counter[RelationType] = Counter()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = tuple(sorted(members, key=lambda item: item.fixture_id))
        anchor = ordered[0]
        relations = tuple(
            RelationType.EXACT_EQUIVALENT
            if anchor.text == member.text
            else RelationType.FORMATTING_EQUIVALENT
            for member in ordered[1:]
        )
        if any(relation not in registry.safe_family_relations for relation in relations):
            raise DevelopmentRedundancyGateError("redundancy_safe_family_classifier_mismatch")
        counts.update(relations)
        families.append(_SafeFamily(ordered, relations))
    return tuple(families), counts


def _relation_counts(counts: Counter[RelationType]) -> tuple[RelationCount, ...]:
    return tuple(RelationCount(relation=relation, count=counts[relation]) for relation in sorted(counts, key=lambda item: item.value))


def evaluate_redundancy_slice(
    slice_input: RedundancySliceInput,
    registry: DevelopmentRedundancyGateRegistry,
) -> SliceRedundancyEvidence:
    records = _load_records(slice_input)
    upstream_owned = tuple(record for record in records if record.metamorphic_relation in registry.upstream_owned_relations)
    evaluated = tuple(record for record in records if record.metamorphic_relation not in registry.upstream_owned_relations)
    families, relation_counts = _safe_families(evaluated, registry)
    family_by_uid = {
        member.fixture_id: family
        for family in families
        for member in family.members
    }
    cross_parent = sum(len({member.parent_record_id for member in family.members}) > 1 for family in families)
    expected_families = recovered_families = expected_copies = linked_copies = 0
    perturbations = perturbation_safe = perturbation_candidates = 0
    if slice_input.scenario == "duplicate_heavy":
        by_parent: dict[str, dict[str, DevelopmentFixtureRecord]] = defaultdict(dict)
        for record in evaluated:
            by_parent[record.parent_record_id][record.metamorphic_relation] = record
        for relations in by_parent.values():
            required = (registry.parent_relation, *registry.exact_copy_relations, registry.perturbation_relation)
            if any(label not in relations for label in required):
                raise DevelopmentRedundancyGateError(f"redundancy_duplicate_fixture_incomplete:{slice_input.slice_id}")
            parent = relations[registry.parent_relation]
            copies = tuple(relations[label] for label in registry.exact_copy_relations)
            perturbation = relations[registry.perturbation_relation]
            expected_families += 1
            expected_copies += len(copies)
            family = family_by_uid.get(parent.fixture_id)
            if family is not None:
                member_ids = {member.fixture_id for member in family.members}
                linked = sum(copy.fixture_id in member_ids for copy in copies)
                linked_copies += linked
                recovered_families += linked == len(copies) and perturbation.fixture_id not in member_ids
            perturbations += 1
            relation = classify_relation(
                RedundancyUnit(parent.fixture_id, parent.text),
                RedundancyUnit(perturbation.fixture_id, perturbation.text),
                registry.settings.to_settings(),
            )
            relation_counts[relation.relation] += 1
            perturbation_safe += relation.safe_family_edge
            perturbation_candidates += relation.relation in registry.candidate_only_relations
    clean_false_merged = 0
    if slice_input.scenario == "clean":
        clean_false_merged = len({member.fixture_id for family in families for member in family.members})
    return SliceRedundancyEvidence(
        slice_id=slice_input.slice_id,
        domain=slice_input.domain,
        scenario=slice_input.scenario,
        artifact_sha256=slice_input.expected_sha256,
        record_count=len(records),
        evaluated_record_count=len(evaluated),
        upstream_owned_record_count=len(upstream_owned),
        safe_family_count=len(families),
        safe_family_member_count=len({member.fixture_id for family in families for member in family.members}),
        cross_parent_safe_family_count=cross_parent,
        expected_exact_family_count=expected_families,
        recovered_exact_family_count=recovered_families,
        expected_exact_copy_count=expected_copies,
        linked_exact_copy_count=linked_copies,
        perturbation_record_count=perturbations,
        perturbation_safe_merge_count=perturbation_safe,
        perturbation_candidate_relation_count=perturbation_candidates,
        clean_false_merged_record_count=clean_false_merged,
        relation_counts=_relation_counts(relation_counts),
    )


def _wilson_upper(failures: int, trials: int, confidence_level: float) -> float:
    if trials <= 0:
        return 1.0
    z = NormalDist().inv_cdf(confidence_level)
    proportion = failures / trials
    denominator = 1 + z * z / trials
    center = proportion + z * z / (2 * trials)
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return min(1.0, (center + radius) / denominator)


def build_development_redundancy_gate(
    registry: DevelopmentRedundancyGateRegistry,
) -> DevelopmentRedundancyGateReport:
    manifest_path = registry.inventory_manifest_path
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file() or _sha256_file(path) != registry.inventory_manifest_file_sha256:
        raise DevelopmentRedundancyGateError("redundancy_inventory_manifest_file_mismatch")
    manifest = DevelopmentCorpusInventoryManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.status is not InventoryStatus.ADMITTED or manifest.manifest_sha256 != registry.inventory_manifest_sha256:
        raise DevelopmentRedundancyGateError("redundancy_inventory_manifest_not_admitted")
    expected = {(domain, scenario) for domain in registry.required_domains for scenario in registry.required_scenarios}
    observed = {(item.domain, item.scenario) for item in manifest.slices}
    matrix_complete = observed == expected and len(manifest.slices) == len(expected)
    if not matrix_complete:
        raise DevelopmentRedundancyGateError("redundancy_development_matrix_incomplete")
    evidence: list[SliceRedundancyEvidence] = []
    for item in manifest.slices:
        if item.status is not SliceStatus.MATERIALIZED or item.artifact_path is None or item.artifact_sha256 is None:
            raise DevelopmentRedundancyGateError(f"redundancy_slice_not_materialized:{item.slice_id}")
        evidence.append(
            evaluate_redundancy_slice(
                RedundancySliceInput(Path(item.artifact_path), item.slice_id, item.domain, item.scenario, item.artifact_sha256),
                registry,
            )
        )
    clean_count = sum(item.evaluated_record_count for item in evidence if item.scenario == "clean")
    clean_false = sum(item.clean_false_merged_record_count for item in evidence)
    perturbation_count = sum(item.perturbation_record_count for item in evidence)
    perturbation_safe = sum(item.perturbation_safe_merge_count for item in evidence)
    clean_upper = _wilson_upper(clean_false, clean_count, registry.confidence_level)
    perturbation_upper = _wilson_upper(perturbation_safe, perturbation_count, registry.confidence_level)
    totals = {
        "expected_families": sum(item.expected_exact_family_count for item in evidence),
        "recovered_families": sum(item.recovered_exact_family_count for item in evidence),
        "expected_copies": sum(item.expected_exact_copy_count for item in evidence),
        "linked_copies": sum(item.linked_exact_copy_count for item in evidence),
        "candidate_perturbations": sum(item.perturbation_candidate_relation_count for item in evidence),
        "cross_parent": sum(item.cross_parent_safe_family_count for item in evidence),
    }
    blockers: list[str] = []
    if totals["expected_families"] == 0 or totals["recovered_families"] != totals["expected_families"]:
        blockers.append("redundancy_exact_family_recall_failed")
    if totals["linked_copies"] != totals["expected_copies"]:
        blockers.append("redundancy_exact_copy_linkage_failed")
    if clean_upper > registry.maximum_clean_false_merge_upper_bound:
        blockers.append("redundancy_clean_false_merge_bound_failed")
    if perturbation_upper > registry.maximum_perturbation_safe_merge_upper_bound:
        blockers.append("redundancy_perturbation_safe_merge_bound_failed")
    if totals["cross_parent"]:
        blockers.append("redundancy_cross_parent_safe_family_detected")
    payload = {
        "schema_version": "development-redundancy-gate-report-v1",
        "registry_sha256": registry.identity_sha256(),
        "inventory_manifest_sha256": manifest.manifest_sha256,
        "inventory_manifest_file_sha256": registry.inventory_manifest_file_sha256,
        "slices": [item.model_dump(mode="json") for item in evidence],
        "matrix_complete": matrix_complete,
        "expected_exact_family_count": totals["expected_families"],
        "recovered_exact_family_count": totals["recovered_families"],
        "expected_exact_copy_count": totals["expected_copies"],
        "linked_exact_copy_count": totals["linked_copies"],
        "clean_control_record_count": clean_count,
        "clean_false_merged_record_count": clean_false,
        "clean_false_merge_upper_bound": clean_upper,
        "perturbation_record_count": perturbation_count,
        "perturbation_safe_merge_count": perturbation_safe,
        "perturbation_safe_merge_upper_bound": perturbation_upper,
        "perturbation_candidate_relation_count": totals["candidate_perturbations"],
        "cross_parent_safe_family_count": totals["cross_parent"],
        "blocker_codes": sorted(blockers),
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "selector_membership_mutated": False,
    }
    return DevelopmentRedundancyGateReport(
        status=RedundancyGateStatus.PASSED if not blockers else RedundancyGateStatus.BLOCKED,
        report_sha256=hash_json(payload),
        **payload,
    )


__all__ = ["build_development_redundancy_gate", "evaluate_redundancy_slice"]
