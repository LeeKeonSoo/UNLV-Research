from __future__ import annotations

import json
from pathlib import Path

from development_corpus_admission_contract import (
    AdmissionStatus,
    BenchmarkContaminationMatch,
    ConfirmatoryDisjointEvidence,
    CorpusBenchmarkScanEvidence,
    CorpusReferenceSpec,
    CorpusRole,
    DevelopmentCorpusAdmissionRegistry,
    DevelopmentCorpusAdmissionError,
    DevelopmentCorpusAdmissionReport,
    FilteredConfirmatoryReference,
    hash_json,
)
from development_corpus_benchmark_exclusion import (
    BenchmarkIndex,
    build_benchmark_index,
    match_benchmark_segments,
    sha256_file,
    token_sha256,
    tokenize,
)
from development_corpus_inventory_contract import InventoryDomain


def _validate_filter_lineage(spec: CorpusReferenceSpec) -> None:
    lineage = spec.filter_lineage
    if lineage is None:
        return
    if sha256_file(Path(lineage.source_path)) != lineage.source_sha256:
        raise DevelopmentCorpusAdmissionError(f"filter_lineage_source_hash_mismatch:{spec.reference_id}")
    evidence_path = Path(lineage.evidence_path)
    if sha256_file(evidence_path) != lineage.evidence_sha256:
        raise DevelopmentCorpusAdmissionError(f"filter_lineage_evidence_hash_mismatch:{spec.reference_id}")
    evidence = FilteredConfirmatoryReference.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    aligned = (
        evidence.reference_id == spec.reference_id
        and evidence.source_path == lineage.source_path
        and evidence.source_sha256 == lineage.source_sha256
        and evidence.output_path == spec.path
        and evidence.output_sha256 == spec.expected_file_sha256
        and evidence.output_record_count == spec.expected_record_count
        and len(evidence.removed_record_ids) == lineage.removed_record_count
    )
    if not aligned:
        raise DevelopmentCorpusAdmissionError(f"filter_lineage_contract_mismatch:{spec.reference_id}")


def _scan_corpus(
    spec: CorpusReferenceSpec,
    role: CorpusRole,
    benchmark_index: BenchmarkIndex,
    width: int,
) -> tuple[CorpusBenchmarkScanEvidence, frozenset[str], frozenset[str], tuple[BenchmarkContaminationMatch, ...]]:
    _validate_filter_lineage(spec)
    path = Path(spec.path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.expected_file_sha256:
        raise DevelopmentCorpusAdmissionError(f"admission_corpus_hash_mismatch:{spec.reference_id}")
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    exact_matches = 0
    containment_matches = 0
    contaminated = 0
    record_count = 0
    traces: list[BenchmarkContaminationMatch] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: JsonValue = json.loads(line)
            if not isinstance(row, dict):
                raise DevelopmentCorpusAdmissionError(f"admission_corpus_row_invalid:{spec.reference_id}")
            text = row.get(spec.text_field)
            if not isinstance(text, str) or not text:
                raise DevelopmentCorpusAdmissionError(f"admission_corpus_text_invalid:{spec.reference_id}")
            record_id = "::".join(str(row[field]) for field in spec.id_fields)
            tokens = tokenize(text)
            text_hash = token_sha256(tokens)
            matches = match_benchmark_segments(tokens, benchmark_index.by_domain[spec.domain], width)
            exact = any(item.match_kind == "exact_text" for item in matches)
            containment = any(item.match_kind == "segment_containment" for item in matches)
            exact_matches += int(exact)
            containment_matches += int(containment)
            contaminated += int(exact or containment)
            record_count += 1
            record_ids.add(record_id)
            text_hashes.add(text_hash)
            traces.extend(
                BenchmarkContaminationMatch(
                    reference_id=spec.reference_id,
                    domain=spec.domain,
                    role=role,
                    record_id=record_id,
                    benchmark_id=item.benchmark_id,
                    match_kind=item.match_kind,
                    segment_sha256=item.segment_sha256,
                    segment_lexical_token_count=item.segment_lexical_token_count,
                )
                for item in matches
            )
    if record_count != spec.expected_record_count or record_count != len(record_ids):
        raise DevelopmentCorpusAdmissionError(f"admission_corpus_identity_mismatch:{spec.reference_id}")
    evidence = CorpusBenchmarkScanEvidence(
        reference_id=spec.reference_id,
        domain=spec.domain,
        role=role,
        record_count=record_count,
        exact_text_match_count=exact_matches,
        segment_containment_match_count=containment_matches,
        contaminated_record_count=contaminated,
    )
    return evidence, frozenset(record_ids), frozenset(text_hashes), tuple(traces)


def build_development_corpus_admission(registry: DevelopmentCorpusAdmissionRegistry) -> DevelopmentCorpusAdmissionReport:
    benchmark_index = build_benchmark_index(registry)
    scanned: list[tuple[CorpusReferenceSpec, CorpusBenchmarkScanEvidence, frozenset[str], frozenset[str], tuple[BenchmarkContaminationMatch, ...]]] = []
    for role, specs in (
        (CorpusRole.DEVELOPMENT, registry.development_sources),
        (CorpusRole.CONFIRMATORY, registry.confirmatory_references),
    ):
        for spec in specs:
            evidence, record_ids, text_hashes, traces = _scan_corpus(
                spec,
                role,
                benchmark_index,
                registry.minimum_containment_segment_lexical_tokens,
            )
            scanned.append((spec, evidence, record_ids, text_hashes, traces))
    development_ids = frozenset().union(*(item[2] for item in scanned if item[1].role is CorpusRole.DEVELOPMENT))
    development_texts = frozenset().union(*(item[3] for item in scanned if item[1].role is CorpusRole.DEVELOPMENT))
    disjoint: list[ConfirmatoryDisjointEvidence] = []
    for spec, scan, record_ids, text_hashes, _traces in scanned:
        if scan.role is not CorpusRole.CONFIRMATORY:
            continue
        disjoint.append(
            ConfirmatoryDisjointEvidence(
                domain=spec.domain,
                reference_id=spec.reference_id,
                source_group_id=spec.source_group_id,
                source_snapshot_id=spec.source_snapshot_id,
                file_sha256=spec.expected_file_sha256,
                record_count=scan.record_count,
                development_record_id_overlap_count=len(record_ids & development_ids),
                development_normalized_text_overlap_count=len(text_hashes & development_texts),
            )
        )
    benchmark_contamination = sum(item[1].contaminated_record_count for item in scanned)
    record_overlap = sum(item.development_record_id_overlap_count for item in disjoint)
    text_overlap = sum(item.development_normalized_text_overlap_count for item in disjoint)
    blockers: list[str] = []
    if benchmark_contamination:
        blockers.append("benchmark_contamination_detected")
    if record_overlap or text_overlap:
        blockers.append("confirmatory_development_overlap_detected")
    frozen_domains = tuple(item.domain for item in disjoint if not item.development_record_id_overlap_count and not item.development_normalized_text_overlap_count)
    benchmark_complete = benchmark_contamination == 0
    payload = {
        "schema_version": "development-corpus-admission-report-v1",
        "registry_sha256": registry.identity_sha256(),
        "benchmark_artifacts": [item.model_dump(mode="json") for item in benchmark_index.evidence],
        "corpus_scans": [item[1].model_dump(mode="json") for item in scanned],
        "confirmatory_references": [item.model_dump(mode="json") for item in disjoint],
        "contamination_matches": [trace.model_dump(mode="json") for item in scanned for trace in item[4]],
        "benchmark_exclusion_complete": benchmark_complete,
        "frozen_confirmatory_domains": [item.value for item in frozen_domains],
        "total_benchmark_contaminated_record_count": benchmark_contamination,
        "total_confirmatory_development_record_id_overlap_count": record_overlap,
        "total_confirmatory_development_text_overlap_count": text_overlap,
        "blocker_codes": sorted(blockers),
        "benchmark_outcomes_read": False,
        "selector_membership_mutated": False,
    }
    return DevelopmentCorpusAdmissionReport(
        status=AdmissionStatus.BLOCKED if blockers else AdmissionStatus.ADMITTED,
        report_sha256=hash_json(payload),
        **payload,
    )


__all__ = ["build_development_corpus_admission"]
