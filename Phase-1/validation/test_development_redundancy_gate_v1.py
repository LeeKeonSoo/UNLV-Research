from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_inventory_contract import InventoryDomain
from development_redundancy_gate import evaluate_redundancy_slice
from development_redundancy_gate_contract import (
    DevelopmentRedundancyGateRegistry,
    DevelopmentRedundancyGateReport,
    RedundancyGateStatus,
    RedundancySettingsSpec,
    RedundancySliceInput,
    load_redundancy_gate_registry,
)
from redundancy_v2 import RelationType


def _normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _record(slice_id: str, parent: str, relation: str, text: str) -> dict[str, str]:
    return {
        "fixture_id": hashlib.sha256(f"{slice_id}:{parent}:{relation}".encode()).hexdigest(),
        "slice_id": slice_id,
        "parent_record_id": parent,
        "metamorphic_relation": relation,
        "text": text,
        "normalized_text_sha256": _normalized_hash(text),
    }


def _registry() -> DevelopmentRedundancyGateRegistry:
    return DevelopmentRedundancyGateRegistry(
        schema_version="development-redundancy-gate-registry-v1",
        status="e2-frozen-redundancy-inputs",
        inventory_manifest_path="configs/development_corpus_manifest_v1.json",
        inventory_manifest_sha256="1" * 64,
        inventory_manifest_file_sha256="2" * 64,
        required_domains=tuple(InventoryDomain),
        required_scenarios=("clean", "duplicate_heavy", "malformed", "boilerplate_heavy", "mixed_raw_like"),
        safe_family_relations=(RelationType.EXACT_EQUIVALENT, RelationType.FORMATTING_EQUIVALENT),
        candidate_only_relations=(
            RelationType.NEAR_SUBSTITUTE,
            RelationType.CONTAINED_PAYLOAD,
            RelationType.SUPERSET_PAYLOAD,
            RelationType.REPEATED_SPAN,
            RelationType.SEMANTIC_DUPLICATE_CANDIDATE,
        ),
        parent_relation="parent-retained-v1",
        exact_copy_relations=("exact-copy-1-v1", "exact-copy-2-v1"),
        perturbation_relation="length-relative-single-token-deletion-v1",
        upstream_owned_relations=("empty-payload-v1", "invalid-utf8-tail-replacement-v1"),
        settings=RedundancySettingsSpec(
            short_exact_only_max_tokens=32,
            near_min_tokens=64,
            near_max_changed_ratio=0.02,
            near_max_changed_tokens=4,
            containment_min_tokens=12,
            repeated_span_min_lexical_tokens=12,
            complementary_overlap_floor=0.18,
        ),
        confidence_level=0.95,
        maximum_clean_false_merge_upper_bound=0.01,
        maximum_perturbation_safe_merge_upper_bound=0.01,
        benchmark_outcomes_available=False,
        utility_available=False,
        selector_membership_mutation_allowed=False,
    )


def _write(path: Path, rows: list[dict[str, str]]) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_copies_form_parent_scoped_families_without_absorbing_perturbations() -> None:
    # Given two parent records with two exact copies and one single-token perturbation each.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "duplicate.jsonl"
        rows: list[dict[str, str]] = []
        for index in range(2):
            parent = f"parent-{index}"
            words = ["alpha" if index == 0 else "beta"] + [
                f"word{chr(97 + position // 26)}{chr(97 + position % 26)}" for position in range(80)
            ]
            text = " ".join(words)
            perturbed = " ".join((*words[:40], *words[41:]))
            rows.extend(
                (
                    _record("code-duplicate_heavy", parent, "parent-retained-v1", text),
                    _record("code-duplicate_heavy", parent, "exact-copy-1-v1", text),
                    _record("code-duplicate_heavy", parent, "exact-copy-2-v1", text),
                    _record("code-duplicate_heavy", parent, "length-relative-single-token-deletion-v1", perturbed),
                )
            )
        digest = _write(path, rows)
        slice_input = RedundancySliceInput(path, "code-duplicate_heavy", InventoryDomain.CODE, "duplicate_heavy", digest)

        # When Redundancy evidence is evaluated.
        evidence = evaluate_redundancy_slice(slice_input, _registry())

        # Then exact copies are linked and perturbed payloads retain candidate-only authority.
        assert evidence.expected_exact_family_count == 2
        assert evidence.recovered_exact_family_count == 2
        assert evidence.expected_exact_copy_count == 4
        assert evidence.linked_exact_copy_count == 4
        assert evidence.perturbation_record_count == 2
        assert evidence.perturbation_safe_merge_count == 0
        assert evidence.perturbation_candidate_relation_count == 2
        assert evidence.cross_parent_safe_family_count == 0


def test_frozen_e2_report_is_hash_linked_and_passed() -> None:
    # Given the frozen E2 registry and report generated from the admitted matrix.
    registry_path = ROOT / "protocols" / "development_redundancy_gate_registry_v1.json"
    report_path = ROOT / "validation" / "frozen_contracts" / "development_redundancy_gate_report_v1.json"
    registry = load_redundancy_gate_registry(registry_path)

    # When the strict report boundary parses the artifact.
    report = DevelopmentRedundancyGateReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    # Then the registry identity and empirical zero-error gates are immutable.
    assert report.status is RedundancyGateStatus.PASSED
    assert report.registry_sha256 == registry.identity_sha256()
    assert report.recovered_exact_family_count == report.expected_exact_family_count == 1200
    assert report.linked_exact_copy_count == report.expected_exact_copy_count == 2400
    assert report.clean_false_merged_record_count == 0
    assert report.perturbation_safe_merge_count == 0
    assert report.cross_parent_safe_family_count == 0
    assert report.blocker_codes == ()


if __name__ == "__main__":
    test_exact_copies_form_parent_scoped_families_without_absorbing_perturbations()
    test_frozen_e2_report_is_hash_linked_and_passed()
    print("[development-redundancy-gate-v1] parent-scoped exact-family behavior: pass")
