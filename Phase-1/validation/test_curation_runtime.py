#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module() -> object:
    path = ROOT / "run_curation.py"
    spec = importlib.util.spec_from_file_location("run_curation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8-sig",
    )


def test_chunk_text_preserves_code_layout_when_a_paragraph_exceeds_limit() -> None:
    module = _load_module()
    source = (
        "def first():\n"
        "    return 'first'\n"
        "def second():\n"
        "    return 'second'\n"
        "def third():\n"
        "    return 'third'\n"
    )

    chunks = module.chunk_text(source, max_chunk_chars=40)

    assert "".join(chunks) == source
    assert any("\n" in chunk for chunk in chunks)
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_chunk_text_preserves_an_unbroken_long_line() -> None:
    module = _load_module()
    source = "x" * 91

    chunks = module.chunk_text(source, max_chunk_chars=40)

    assert "".join(chunks) == source
    assert [len(chunk) for chunk in chunks] == [40, 40, 11]


def main() -> int:
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        input_path = work_dir / "candidates.jsonl"
        output_dir = work_dir / "curated"
        config_path = work_dir / "contract.json"
        audit_path = work_dir / "benchmark_exclusion_audit.json"
        _write_jsonl(
            input_path,
            [
                {
                    "id": "code-1",
                    "content": "def add(left, right):\n    return left + right\n\nThis implementation returns the sum without side effects.",
                    "pii_context": "repository_code",
                    "language": {"code": "python", "version": "3.11", "confidence": 1.0},
                    "artifact_context": {"generation": "authored", "dependency_copy": False},
                    "partition": {"content_type": "code", "path": "src/example.py"},
                },
                {
                    "uid": "math-1",
                    "document": "Theorem. For every differentiable function, the derivative of a sum is the sum of its derivatives. Proof. Apply the limit definition to each term.",
                },
                {
                    "id": "multilingual-1",
                    "body": "이 연구 보고서는 실험 절차와 관측 결과를 독립적으로 검토할 수 있도록 충분히 자세히 설명한다.",
                },
            ],
        )
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "normal",
                    "input": {
                        "candidate_files": [str(input_path)],
                        "text_fields": ["text", "content", "document", "body"],
                        "defaults": {},
                    },
                    "output_dir": str(output_dir),
                    "pretraining_audit_path": str(audit_path),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {
                        "minimum_residual_chars": 40,
                        "no_binding_budget_action": "selection_without_binding_budget",
                    },
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8-sig",
        )
        audit_path.write_text(
            json.dumps(
                {
                    "status": "benchmark_exclusion_complete",
                    "pretraining_eligible": True,
                    "audited_output": {
                        "path": str(input_path),
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )

        module = _load_module()
        report = module.materialize(config_path)

        assert report["summary"]["input_records"] == 3
        assert report["summary"]["stage_a_release_records"] == 3
        assert report["summary"]["stage_c_curated_chunks"] == 3
        assert report["summary"]["stage_c_near_duplicate_removed_chunks"] == 0
        assert report["summary"]["stage_c_structural_scaffold_removed_chunks"] == 0
        assert report["stage_contract"]["stage_a"] == "source_agnostic_text_normalization_and_integrity_handling"
        assert report["stage_contract"]["stage_b"] == "chunk_level_hard_gate"
        assert report["stage_contract"]["stage_c"] == "reason_coded_redundancy_and_quality_retention_without_implicit_budget"
        assert report["summary"]["stage_c_explicit_non_payload_rejected_chunks"] == 0
        assert report["summary"]["stage_c_positive_quality_kept_chunks"] == 0
        assert report["summary"]["stage_c_quality_abstain_retained_chunks"] == 3
        assert report["curation_mode"]["mode"] == "normal"
        assert report["curation_mode"]["profile_id"] == "normal_structural_v1"
        assert len(report["curation_mode"]["effective_policy_sha256"]) == 64
        assert report["effective_policy_manifest"]["profile_id"] == "normal_structural_v1"
        assert report["effective_policy_manifest"]["policy"] == report["curation_mode"]["effective_policy"]
        composition = report["composition_audit"]
        assert composition["authority"] == "audit_only"
        assert composition["stages"]["raw_input"]["content_domain"]["records"]["code"] == 1
        assert composition["stages"]["raw_input"]["content_domain"]["records"]["mathematics"] == 1
        assert composition["stages"]["stage_c_curated"]["language_script"]["records"]["non_latin"] == 1
        assert "code" in composition["delta_from_raw"]["stage_c_curated"]["content_domain"]["token_share"]
        reason_impact = report["reason_code_impact_audit"]
        assert reason_impact["authority"] == "audit_only"
        assert reason_impact["selector_consumes_this_audit"] is False
        assert set(reason_impact["stages"]) == {"stage_a_quarantine", "stage_b_rejection", "stage_c_compaction"}
        assert report["coverage_impact_audit"]["selector_consumes_this_audit"] is False
        assert report["coverage_impact_audit"]["authority"] == "materialization_invariant"
        assert report["measurement_contract"]["runtime_token_measurement"] == "whitespace_proxy_non_training"
        assert report["measurement_contract"]["exact_tokenizer_count"] is None
        assert "stage_c_curated_whitespace_token_proxy" in report["summary"]
        assert "stage_c_curated_token_proxy" not in report["summary"]
        assert report["pretraining_audit"]["status"] == "benchmark_exclusion_complete"
        foundation = report["framework_runtime"]
        assert foundation["schema_version"] == "framework-runtime-foundation-report-v1"
        assert foundation["bridge_status"] == "runtime_integrated_block_7"
        assert foundation["new_v1_policy_activation"] is False
        assert foundation["blocked_v1_policy_ids"] == [
            "redundancy.symmetric_near_duplicate_candidate",
            "quality.teacher_panel_candidate",
        ]
        assert [
            (ticket["stage_id"], ticket["core_id"])
            for ticket in foundation["stage_tickets"]
        ] == [
            ("stage_a", "validity"),
            ("stage_b", "redundancy"),
            ("stage_b", "quality"),
            ("stage_c", "coverage"),
        ]
        assert set(report["policy_fingerprint"]["runtime_modules"]) == {
            "framework_objects.py",
            "framework_profiles.py",
            "framework_runtime_bridge.py",
            "hard_structural_runtime.py",
            "general_web_span_compaction.py",
            "ingestion/input_adapter.py",
            "ingestion/candidate_processing.py",
            "inline_license_comment_block_compaction.py",
            "inline_license_header_compaction.py",
            "quality_decision_contract.py",
            "quality_rule_evidence.py",
            "quality_retention.py",
            "run_curation.py",
            "span_level_template_compaction.py",
            "stage_c_selection.py",
            "stage_permissions.py",
        }
        assert len(report["policy_fingerprint"]["runtime_modules"]["run_curation.py"]) == 64
        assert (output_dir / "stage_c_curated_chunks.jsonl").is_file()
        curated = [json.loads(line) for line in (output_dir / "stage_c_curated_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        code_chunk = next(row for row in curated if row["chunk_uid"].startswith("code-1"))
        assert code_chunk["stage_b_decision"]["trigger"] == "no_stage_b_hard_gate_reason"
        assert code_chunk["stage_c_selection"]["trigger"] == "no_symmetric_near_duplicate_match"
        assert code_chunk["quality_retention_decision"]["decision"] == "abstain_retain"
        assert code_chunk["quality_retention_decision"]["schema_version"] == "quality-retention-decision-v2"
        assert code_chunk["quality_retention_decision"]["routing_precondition"]["quality_evidence"] is False
        assert code_chunk["stage_c_policy_metadata"] == {}
        assert code_chunk["stage_c_selector_visible"]["source_name"] is False
        assert code_chunk["stage_c_selector_visible"]["declared_artifact_context"] is False
        assert code_chunk["token_proxy_kind"] == "whitespace_proxy_non_training"

        duplicate_rows = [
            {
                "record_id": "z-record",
                "text": "same payload\n",
                "provenance": {"source_name": "fixture"},
                "rights": {"license": "fixture"},
                "composition": {},
                "language": {"code": "und"},
            },
            {
                "record_id": "a-record",
                "text": "same payload\n",
                "provenance": {"source_name": "fixture"},
                "rights": {"license": "fixture"},
                "composition": {},
                "language": {"code": "und"},
            },
        ]
        stage_b_policy = {
            "max_chunk_chars": 6000,
            "deduplicate_stage_a_text_exactly": True,
        }
        first_passed, first_rejected = module._stage_b_chunks(duplicate_rows, stage_b_policy, text_only=True)
        second_passed, second_rejected = module._stage_b_chunks(
            list(reversed(duplicate_rows)), stage_b_policy, text_only=True
        )
        assert [row["chunk_uid"] for row in first_passed] == ["a-record::0000"]
        assert [row["chunk_uid"] for row in second_passed] == ["a-record::0000"]
        assert first_rejected[0]["stage_b_decision"]["representative_chunk_uid"] == "a-record::0000"
        assert second_rejected[0]["stage_b_decision"]["representative_chunk_uid"] == "a-record::0000"

    print("[curation-runtime] generic cross-domain materialization: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
