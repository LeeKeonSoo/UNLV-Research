#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from quality_teacher_fixtures import (
    build_behavior_fixture_matrix,
    build_protected_fixture_set,
    build_ranker_enrichment_fixture_set,
)


def _verifier_mapping(fixture: Any) -> dict[str, str] | None:
    verifier = fixture.unit.declared_verifier
    return None if verifier is None else verifier.model_dump(mode="json")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(output_dir: Path) -> dict[str, Any]:
    behavior = build_behavior_fixture_matrix(samples_per_cell=8)
    ranker_enrichment = build_ranker_enrichment_fixture_set(samples_per_cell=12)
    protected = build_protected_fixture_set(samples_per_route=200)
    behavior_path = output_dir / "behavior_512.jsonl"
    enrichment_path = output_dir / "ranker_enrichment_576.jsonl"
    protected_path = output_dir / "protected_controlled_800.jsonl"
    behavior_count = _write_jsonl(
        behavior_path,
        (
            {
                "chunk_uid": fixture.fixture_id,
                "uid": fixture.fixture_id,
                "text": fixture.unit.text,
                "quality_declared_context": fixture.unit.declared_context,
                "quality_attached_evidence": list(fixture.unit.attached_evidence),
                "fixture_policy_id": fixture.policy_id,
                "fixture_route": fixture.route,
                "fixture_class": fixture.fixture_class.value,
                "expected_decision": fixture.expected_decision,
                "expected_reason_code": fixture.expected_reason_code,
                "label_provenance": fixture.label_provenance,
            }
            for fixture in behavior
        ),
    )
    enrichment_count = _write_jsonl(
        enrichment_path,
        (
            {
                "chunk_uid": fixture.fixture_id,
                "uid": fixture.fixture_id,
                "text": fixture.unit.text,
                "quality_declared_context": fixture.unit.declared_context,
                "quality_attached_evidence": list(fixture.unit.attached_evidence),
                "quality_declared_verifier": _verifier_mapping(fixture),
                "fixture_policy_id": fixture.policy_id,
                "fixture_route": fixture.route,
                "fixture_class": fixture.fixture_class.value,
                "expected_decision": fixture.expected_decision,
                "expected_reason_code": fixture.expected_reason_code,
                "label_provenance": fixture.label_provenance,
                "ranker_training_authority": "target_policy_only",
            }
            for fixture in ranker_enrichment
        ),
    )
    protected_count = _write_jsonl(
        protected_path,
        (
            {
                "chunk_uid": fixture.fixture_id,
                "uid": fixture.fixture_id,
                "text": fixture.unit.text,
                "quality_declared_context": fixture.unit.declared_context,
                "quality_attached_evidence": list(fixture.unit.attached_evidence),
                "fixture_route": fixture.route,
                "expected_quality_gate": fixture.expected_quality_gate,
                "label_provenance": "deterministic_construction",
            }
            for fixture in protected
        ),
    )
    audit = {
        "schema_version": "quality-teacher-fixture-dataset-audit-v2",
        "behavior_path": str(behavior_path),
        "behavior_sha256": _sha256(behavior_path),
        "behavior_count": behavior_count,
        "behavior_expected_counts": dict(
            sorted(Counter(fixture.expected_decision for fixture in behavior).items())
        ),
        "ranker_enrichment_path": str(enrichment_path),
        "ranker_enrichment_sha256": _sha256(enrichment_path),
        "ranker_enrichment_count": enrichment_count,
        "ranker_enrichment_expected_counts": dict(
            sorted(Counter(fixture.expected_decision for fixture in ranker_enrichment).items())
        ),
        "ranker_enrichment_unique_uid_count": len(
            {fixture.fixture_id for fixture in ranker_enrichment}
        ),
        "ranker_enrichment_unique_text_count": len(
            {fixture.unit.text for fixture in ranker_enrichment}
        ),
        "ranker_training_authority": "target_policy_only",
        "protected_path": str(protected_path),
        "protected_sha256": _sha256(protected_path),
        "protected_count": protected_count,
        "label_provenance": "deterministic_construction",
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "source_reputation_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize deterministic Quality fixtures")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir.resolve()), ensure_ascii=True))


if __name__ == "__main__":
    main()
