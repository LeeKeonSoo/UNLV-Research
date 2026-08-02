from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_admission import build_development_corpus_admission
from development_corpus_admission_alignment import validate_admission_inventory_alignment
from development_confirmatory_filter import filter_confirmatory_reference
from development_corpus_admission_contract import (
    AdmissionStatus,
    BenchmarkArtifactFormat,
    BenchmarkArtifactSpec,
    CorpusReferenceSpec,
    DevelopmentCorpusAdmissionRegistry,
    FilterLineageSpec,
)
from development_corpus_inventory_contract import (
    ConfirmatoryReference,
    DevelopmentCorpusInventoryRegistry,
    InventoryDomain,
    InventorySourceSpec,
)


def _write_jsonl(path: Path, rows: list[dict[str, str | list[str]]]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, prefix: str, contaminating_text: str | None = None) -> str:
    rows = [
        {
            "record_id": f"{prefix}-{index}",
            "text": contaminating_text if index == 0 and contaminating_text else f"{prefix} independent payload number {index} with enough distinct words",
        }
        for index in range(3)
    ]
    return _write_jsonl(path, rows)


def _registry(
    root: Path,
    contamination: bool = False,
    overlap: bool = False,
    confirmatory_contamination: bool = False,
) -> DevelopmentCorpusAdmissionRegistry:
    development_sources: list[CorpusReferenceSpec] = []
    confirmatory_sources: list[CorpusReferenceSpec] = []
    benchmarks: list[BenchmarkArtifactSpec] = []
    benchmark_text = "calculate the exact answer for this deliberately frozen benchmark prompt using all supplied values and constraints"
    for domain in InventoryDomain:
        benchmark_path = root / f"{domain.value}-benchmark.jsonl"
        benchmark_hash = _write_jsonl(benchmark_path, [{"task_id": "0", "segments": [benchmark_text]}])
        benchmarks.append(
            BenchmarkArtifactSpec(
                benchmark_id=f"{domain.value}-benchmark",
                domain=domain,
                path=str(benchmark_path),
                expected_file_sha256=benchmark_hash,
                artifact_format=BenchmarkArtifactFormat.JSONL,
            )
        )
        development_path = root / f"{domain.value}-development.jsonl"
        development_hash = _source(
            development_path,
            f"{domain.value}-development",
            benchmark_text if contamination and domain is InventoryDomain.MATH else None,
        )
        development_sources.append(
            CorpusReferenceSpec(
                reference_id=f"{domain.value}-development",
                domain=domain,
                source_group_id=f"{domain.value}-development-group",
                source_snapshot_id=f"{domain.value}-development-snapshot",
                path=str(development_path),
                id_fields=("record_id",),
                text_field="text",
                expected_file_sha256=development_hash,
                expected_record_count=3,
                selector_visible_source_metadata=False,
            )
        )
        confirmatory_path = root / f"{domain.value}-confirmatory.jsonl"
        confirmatory_hash = _source(
            confirmatory_path,
            f"{domain.value}-confirmatory",
            benchmark_text if confirmatory_contamination and domain is InventoryDomain.MATH else None,
        )
        if overlap and domain is InventoryDomain.GENERAL:
            confirmatory_path.write_bytes(development_path.read_bytes())
            confirmatory_hash = hashlib.sha256(confirmatory_path.read_bytes()).hexdigest()
        confirmatory_sources.append(
            CorpusReferenceSpec(
                reference_id=f"{domain.value}-confirmatory",
                domain=domain,
                source_group_id=f"{domain.value}-confirmatory-group",
                source_snapshot_id=f"{domain.value}-confirmatory-snapshot",
                path=str(confirmatory_path),
                id_fields=("record_id",),
                text_field="text",
                expected_file_sha256=confirmatory_hash,
                expected_record_count=3,
                selector_visible_source_metadata=False,
            )
        )
    return DevelopmentCorpusAdmissionRegistry(
        schema_version="development-corpus-admission-registry-v1",
        status="e1-frozen-admission-inputs",
        normalization="unicode-nfkc-casefold-whitespace-token-v1",
        minimum_exact_segment_lexical_tokens=8,
        minimum_containment_segment_lexical_tokens=13,
        development_sources=tuple(development_sources),
        confirmatory_references=tuple(confirmatory_sources),
        benchmark_artifacts=tuple(benchmarks),
        benchmark_outcomes_available=False,
        selector_membership_mutation_allowed=False,
    )


def test_admission_accepts_disjoint_sources_without_benchmark_contamination() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = build_development_corpus_admission(_registry(Path(directory)))
        assert report.status is AdmissionStatus.ADMITTED
        assert report.blocker_codes == ()
        assert report.benchmark_exclusion_complete is True
        assert report.frozen_confirmatory_domains == tuple(InventoryDomain)
        assert report.total_benchmark_contaminated_record_count == 0
        assert report.total_confirmatory_development_text_overlap_count == 0
        assert report.benchmark_outcomes_read is False
        assert report.selector_membership_mutated is False


def test_admission_blocks_exact_benchmark_prompt_containment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = build_development_corpus_admission(_registry(Path(directory), contamination=True))
        assert report.status is AdmissionStatus.BLOCKED
        assert "benchmark_contamination_detected" in report.blocker_codes
        assert report.total_benchmark_contaminated_record_count == 1


def test_admission_blocks_confirmatory_development_overlap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = build_development_corpus_admission(_registry(Path(directory), overlap=True))
        assert report.status is AdmissionStatus.BLOCKED
        assert "confirmatory_development_overlap_detected" in report.blocker_codes
        assert report.total_confirmatory_development_text_overlap_count == 3


def test_alignment_rejects_inventory_source_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        admission = _registry(Path(directory))
        sources = tuple(
            InventorySourceSpec(
                source_id=item.reference_id,
                domain=item.domain,
                role="raw_like",
                path=item.path,
                id_fields=item.id_fields,
                text_field=item.text_field,
                expected_file_sha256=item.expected_file_sha256,
                selector_visible_source_metadata=False,
            )
            for item in admission.development_sources
        )
        inventory = DevelopmentCorpusInventoryRegistry(
            schema_version="development-corpus-inventory-registry-v1",
            status="block-8-inventory-only",
            normalization="unicode-nfkc-whitespace-collapse-v1",
            output_root=str(Path(directory) / "matrix"),
            parent_records_per_slice=1,
            sources=sources + tuple(
                item.model_copy(update={"source_id": item.source_id + "-clean", "role": "clean_control"})
                for item in sources
            ),
            confirmatory_references={domain: ConfirmatoryReference.FROZEN for domain in InventoryDomain},
            metamorphic_transformations={
                "duplicate_heavy": ("duplicate-v1",),
                "malformed": ("malformed-v1",),
                "boilerplate_heavy": ("boilerplate-v1",),
            },
            benchmark_outcomes_available=False,
            selector_membership_mutation_allowed=False,
        )
        try:
            validate_admission_inventory_alignment(admission, inventory)
        except ValueError as error:
            assert str(error) == "admission_inventory_source_ids_mismatch"
        else:
            raise AssertionError("An incomplete admission registry aligned with the inventory")


def test_filter_materializes_a_traceable_confirmatory_view() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _registry(root, confirmatory_contamination=True)
        report = build_development_corpus_admission(registry)
        evidence = filter_confirmatory_reference(registry, report, InventoryDomain.MATH, root / "filtered.jsonl")
        assert evidence.input_record_count == 3
        assert evidence.output_record_count == 2
        assert evidence.removed_record_ids == ("math-confirmatory-0",)
        assert Path(evidence.output_path).is_file()
        evidence_path = root / "filtered.evidence.json"
        evidence_path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
        updated_references = tuple(
            item.model_copy(
                update={
                    "path": evidence.output_path,
                    "expected_file_sha256": evidence.output_sha256,
                    "expected_record_count": evidence.output_record_count,
                    "source_snapshot_id": f"filtered-{evidence.output_sha256}",
                    "filter_lineage": FilterLineageSpec(
                        source_path=evidence.source_path,
                        source_sha256=evidence.source_sha256,
                        evidence_path=str(evidence_path),
                        evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                        removed_record_count=1,
                    ),
                }
            )
            if item.domain is InventoryDomain.MATH
            else item
            for item in registry.confirmatory_references
        )
        admitted = build_development_corpus_admission(
            registry.model_copy(update={"confirmatory_references": updated_references})
        )
        assert admitted.status is AdmissionStatus.ADMITTED


if __name__ == "__main__":
    test_admission_accepts_disjoint_sources_without_benchmark_contamination()
    test_admission_blocks_exact_benchmark_prompt_containment()
    test_admission_blocks_confirmatory_development_overlap()
    test_alignment_rejects_inventory_source_drift()
    test_filter_materializes_a_traceable_confirmatory_view()
    print("[development-corpus-admission-v1] disjoint admission and fail-closed contamination: pass")
