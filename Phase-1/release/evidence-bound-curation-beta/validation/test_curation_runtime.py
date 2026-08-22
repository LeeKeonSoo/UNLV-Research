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

from quality_model_evidence import QualityDecision, QualityPolicyEvidence
from model_provider_contract import load_provider_registry
from runtime_artifact_materialization import RuntimeArtifactBundle
from semantic_coverage_materializer import SemanticCoverageMaterializationError


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _all_pass_quality_scorer(rows, **_kwargs):
    results = {}
    for row in rows:
        uid = str(row["chunk_uid"])
        policy_results = []
        for policy_id in QUALITY_POLICY_IDS:
            policy_results.append(
                QualityPolicyEvidence(
                    policy_id=policy_id,
                    decision=QualityDecision.PASS,
                    reason_codes=("quality_ranker_pass",),
                    class_probabilities=(("pass", 1.0), ("fail", 0.0)),
                    failure_probability=0.0,
                    failure_threshold=0.7,
                    prediction_confidence=1.0,
                    minimum_decision_confidence=0.7,
                    out_of_distribution=False,
                    ranker_artifact_sha256="a" * 64,
                )
            )
        results[uid] = tuple(policy_results)
    return results, {"fixture": True, "input_chunks": len(rows)}


def _load_module() -> object:
    path = ROOT / "run_curation.py"
    spec = importlib.util.spec_from_file_location("run_curation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_runtime_output_preflight_fails_before_expensive_work() -> None:
    module = _load_module()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        blocking_file = root / "not-a-directory"
        blocking_file.write_text("fixture", encoding="utf-8")

        try:
            module.preflight_runtime_writes(
                output_dir=root / "output",
                checkpoint_path=blocking_file / "checkpoint.json",
            )
        except module.RuntimeWritePreflightError as error:
            assert error.path == blocking_file
        else:
            raise AssertionError("Expected output preflight to reject the blocked path")


def test_semantic_artifact_preflight_runs_before_quality_scoring() -> None:
    module = _load_module()
    quality_called = False

    def forbidden_quality_scorer(rows, **_kwargs):
        nonlocal quality_called
        quality_called = True
        return _all_pass_quality_scorer(rows)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.jsonl"
        config_path = root / "config.json"
        _write_jsonl(
            input_path,
            [{"id": "fixture", "text": "A substantive fixture payload remains valid."}],
        )
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "framework",
                    "execution_scope": "development",
                    "input": {
                        "candidate_files": [str(input_path)],
                        "text_fields": ["text"],
                        "defaults": {},
                    },
                    "output_dir": str(root / "output"),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {
                        "minimum_residual_chars": 40,
                        "no_binding_budget_action": "selection_without_binding_budget",
                        "semantic_coverage": {
                            "provider_registry_path": str(
                                ROOT / "configs" / "model_provider_registry_v1.json"
                            ),
                            "provider_id": "qwen3-embedding-0.6b-semantic-candidate",
                            "corpus_path": str(root / "missing-corpus.jsonl"),
                            "graph_path": str(root / "missing-graph.json"),
                        },
                    },
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8",
        )
        try:
            module.materialize(config_path, quality_scorer=forbidden_quality_scorer)
        except SemanticCoverageMaterializationError as error:
            assert "missing" in str(error).lower()
        else:
            raise AssertionError("Missing Stage-C artifacts must stop the run")

    assert quality_called is False


def test_auto_runtime_artifacts_are_consumed_by_quality_and_coverage() -> None:
    module = _load_module()
    artifact_calls = 0

    def fake_artifact_materializer(request):
        nonlocal artifact_calls
        artifact_calls += 1
        request.output_root.mkdir(parents=True, exist_ok=True)
        corpus_path = request.output_root / "corpus.jsonl"
        corpus_path.write_text(
            "".join(
                json.dumps({"uid": row.chunk_uid, "text": row.text}) + "\n"
                for row in request.universe
            ),
            encoding="utf-8",
        )
        registry = load_provider_registry(request.provider_registry)
        provider_id = request.providers["primary"].provider_id
        provider = next(item for item in registry.providers if item.provider_id == provider_id)
        graph_path = request.output_root / "semantic_coverage_graph.json"
        graph_path.write_text(
            json.dumps(
                {
                    "schema_version": "semantic-coverage-graph-v1",
                    "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                    "graph_sha256": "3" * 64,
                    "primary_provider_id": provider_id,
                    "primary_provider_identity_sha256": provider.identity_sha256(),
                    "stable_strata": [[row.chunk_uid for row in request.universe]],
                    "uncertain_strata": [],
                    "similarities": [],
                }
            ),
            encoding="utf-8",
        )
        embedding = request.output_root / "primary" / "embedding_manifest.json"
        embedding.parent.mkdir(parents=True, exist_ok=True)
        embedding.write_text("{}", encoding="utf-8")
        audit = request.output_root / "semantic_coverage_empirical_audit.json"
        audit.write_text("{}", encoding="utf-8")
        return RuntimeArtifactBundle(
            source_path=request.output_root / "stage_b_universe.jsonl",
            quality_embedding_manifest=embedding,
            coverage_corpus=corpus_path,
            coverage_graph=graph_path,
            coverage_audit=audit,
            primary_provider_id=provider_id,
        )

    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.jsonl"
        output_dir = root / "output"
        config_path = root / "config.json"
        _write_jsonl(
            input_path,
            [
                {"id": "a", "text": "A complete first payload with meaningful relations."},
                {"id": "b", "text": "A complete second payload with meaningful relations."},
            ],
        )
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "framework",
                    "execution_scope": "development",
                    "input": {
                        "candidate_files": [str(input_path)],
                        "text_fields": ["text"],
                        "defaults": {},
                    },
                    "output_dir": str(output_dir),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {
                        "minimum_residual_chars": 40,
                        "no_binding_budget_action": "selection_without_binding_budget",
                    },
                    "runtime_artifacts": {
                        "mode": "auto",
                        "cache_dir": str(root / "cache"),
                        "provider_registry": str(
                            ROOT / "configs" / "model_provider_registry_v1.json"
                        ),
                        "providers": {
                            "primary": {
                                "provider_id": "qwen3-embedding-0.6b-semantic-candidate",
                                "pooling": "last_token",
                                "max_length": 128,
                                "batch_size": 2,
                                "device": "cpu",
                                "append_eos": False,
                            },
                            "audit": {
                                "provider_id": "bge-m3-semantic-audit-candidate",
                                "pooling": "cls",
                                "max_length": 128,
                                "batch_size": 2,
                                "device": "cpu",
                                "append_eos": False,
                            },
                        },
                        "neighbor_count": 1,
                        "block_size": 8,
                        "graph_device": "cpu",
                    },
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8",
        )

        report = module.materialize(
            config_path,
            quality_scorer=_all_pass_quality_scorer,
            artifact_materializer=fake_artifact_materializer,
        )

    assert artifact_calls == 1
    assert report["stage_c_coverage"]["semantic_graph_consumed"] is True
    assert report["summary"]["stage_c_curated_chunks"] == 2
    assert report["runtime_artifacts"]["materialized"] is True
    assert report["runtime_artifacts"]["cache_hit"] is False


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
                    "curation_mode": "framework",
                    "execution_scope": "development",
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
        report = module.materialize(
            config_path, quality_scorer=_all_pass_quality_scorer
        )

        assert report["summary"]["input_records"] == 3
        assert report["summary"]["stage_a_release_records"] == 3
        assert report["summary"]["stage_c_curated_chunks"] == 3
        assert report["summary"]["stage_b_near_duplicate_removed_chunks"] == 0
        assert report["summary"]["stage_b_structural_scaffold_removed_chunks"] == 0
        assert report["stage_contract"]["stage_a"] == "source_agnostic_text_normalization_and_integrity_handling"
        assert report["stage_contract"]["stage_b"] == (
            "redundancy_removal_and_explicit_quality_failure_filtering"
        )
        assert report["stage_contract"]["stage_c"] == "coverage_veto_and_final_materialization"
        assert report["summary"]["stage_b_explicit_non_payload_rejected_chunks"] == 0
        assert report["summary"]["stage_b_quality_not_selected_chunks"] == 0
        assert report["stage_b_quality"]["retained_chunks"] == 3
        assert report["curation_mode"]["mode"] == "framework"
        assert report["curation_mode"]["profile_id"] == "framework_structural_v2"
        assert len(report["curation_mode"]["effective_policy_sha256"]) == 64
        assert report["effective_policy_manifest"]["profile_id"] == "framework_structural_v2"
        assert report["effective_policy_manifest"]["policy"] == report["curation_mode"]["effective_policy"]
        composition = report["composition_audit"]
        assert composition["authority"] == "audit_only"
        assert composition["stages"]["raw_input"]["content_domain"]["records"]["code"] == 1
        assert composition["stages"]["raw_input"]["content_domain"]["records"]["mathematics"] == 1
        assert composition["stages"]["stage_c_curated"]["language_script"]["records"]["non_latin"] == 1
        assert "stage_c_curated" not in composition["delta_from_raw"]
        assert composition["excluded_cross_unit_deltas"]["stage_c_curated"] == "record_to_chunk_delta_not_emitted"
        assert "code" in composition["delta_from_stage_b_pass"]["stage_c_curated"]["content_domain"]["token_share"]
        explanatory = report["composition_artifacts"]
        assert explanatory["authority"] == "audit_only"
        assert explanatory["consumed_by_selection"] is False
        assert explanatory["target_distribution_enforced"] is False
        for name in (
            "composition_audit.json",
            "composition_by_route.csv",
            "composition_by_language.csv",
            "eligible_curated_composition_delta.csv",
        ):
            assert (output_dir / name).is_file()
        reason_impact = report["reason_code_impact_audit"]
        assert reason_impact["authority"] == "audit_only"
        assert reason_impact["selector_consumes_this_audit"] is False
        assert set(reason_impact["stages"]) == {
            "stage_a_quarantine",
            "stage_a_chunk_rejection",
            "stage_b_rejection",
            "stage_b_policy_removal",
            "stage_b_span_transformation",
        }
        assert report["coverage_impact_audit"]["selector_consumes_this_audit"] is False
        assert report["coverage_impact_audit"]["authority"] == "materialization_invariant"
        assert report["measurement_contract"]["runtime_token_measurement"] == "whitespace_proxy_non_training"
        assert report["measurement_contract"]["exact_tokenizer_count"] is None
        assert "stage_c_curated_whitespace_token_proxy" in report["summary"]
        assert "stage_c_curated_token_proxy" not in report["summary"]
        assert report["pretraining_audit"]["status"] == "benchmark_exclusion_complete"
        foundation = report["framework_runtime"]
        assert foundation["schema_version"] == "runtime-stage-authority-report-v1"
        assert len(foundation["framework_manifest_sha256"]) == 64
        assert len(foundation["policy_registry_sha256"]) == 64
        assert len(foundation["policy_profiles_sha256"]) == 64
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
            "composition_audit.py",
            "composition_artifacts.py",
            "all_policy_stage_b.py",
            "content_router.py",
            "coverage_contract.py",
            "coverage_engine.py",
            "coverage_metrics.py",
            "coverage_rematerialization.py",
            "coverage_taxonomy.py",
            "coverage_redundancy_bridge.py",
            "curation_artifacts.py",
            "model_provider_contract.py",
            "ingestion/input_adapter.py",
            "ingestion/candidate_processing.py",
            "quality_decision_contract.py",
            "quality_rule_evidence.py",
            "quality_retention.py",
            "quality_fallback_evidence.py",
            "reason_code_audit.py",
            "repeated_sentence_compaction.py",
            "redundancy_checkpoint.py",
            "redundancy_equivalence.py",
            "redundancy_mode_policy.py",
            "redundancy_v2.py",
            "redundancy_v2_retrieval.py",
            "runtime_artifact_materialization.py",
            "quality_model_evidence.py",
            "quality_ranker_artifact.py",
            "quality_ranker_policy.py",
            "quality_ranker_runtime.py",
            "quality_operating_points.py",
            "quality_stage_bridge.py",
            "quality_teacher_observation_codec.py",
            "run_curation.py",
            "stage_b_policy.py",
            "stage_permissions.py",
            "semantic_coverage_bundle.py",
            "semantic_coverage_corpus_runner.py",
            "semantic_coverage_empirical_audit.py",
            "semantic_coverage_graph.py",
            "semantic_coverage_materializer.py",
            "semantic_embedding_artifact.py",
            "semantic_embedding_runtime.py",
            "semantic_neighbor_runtime.py",
        }
        assert len(report["policy_fingerprint"]["runtime_modules"]["run_curation.py"]) == 64
        assert (output_dir / "stage_c_curated_chunks.jsonl").is_file()
        curated = [json.loads(line) for line in (output_dir / "stage_c_curated_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        code_chunk = next(row for row in curated if row["chunk_uid"].startswith("code-1"))
        assert code_chunk["stage_a_decision"]["trigger"] == "stage_a_chunk_integrity_pass"
        assert code_chunk["stage_b_decision"]["trigger"] == "normalized_text_digest_is_distinct"
        assert code_chunk["stage_b_policy"]["trigger"] == "no_structural_nonpayload_evidence"
        assert code_chunk["quality_retention_decision"]["decision"] == "abstain_retain"
        assert code_chunk["quality_retention_decision"]["schema_version"] == "quality-retention-decision-v2"
        assert code_chunk["quality_retention_decision"]["routing_precondition"]["quality_evidence"] is False
        assert code_chunk["stage_b_redundancy_v2"]["action"] == "retain"
        assert len(code_chunk["quality_policy_evidence"]) == 4
        assert code_chunk["quality_stage_decision"]["stage_b_action"] == "retain"
        assert code_chunk["quality_stage_decision"]["stage_b_reason_code"] == (
            "quality_local_positive_support"
        )
        assert code_chunk["quality_stage_decision"]["decision_source"] == (
            "distilled_ranker"
        )
        assert code_chunk["stage_b_policy_metadata"] == {}
        assert code_chunk["stage_b_selector_visible"]["source_name"] is False
        assert code_chunk["stage_b_selector_visible"]["declared_artifact_context"] is False
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
        first_stage_a, first_stage_a_rejected = module._stage_a_chunks(
            duplicate_rows, max_chunk_chars=6000, text_only=True
        )
        second_stage_a, second_stage_a_rejected = module._stage_a_chunks(
            list(reversed(duplicate_rows)), max_chunk_chars=6000, text_only=True
        )
        assert not first_stage_a_rejected
        assert not second_stage_a_rejected
        first_passed, first_rejected = module._stage_b_exact_duplicates(
            first_stage_a, enabled=True
        )
        second_passed, second_rejected = module._stage_b_exact_duplicates(
            second_stage_a, enabled=True
        )
        assert [row["chunk_uid"] for row in first_passed] == ["a-record::0000"]
        assert [row["chunk_uid"] for row in second_passed] == ["a-record::0000"]
        assert first_rejected[0]["stage_b_decision"]["representative_chunk_uid"] == "a-record::0000"
        assert second_rejected[0]["stage_b_decision"]["representative_chunk_uid"] == "a-record::0000"

        passed = [
            {"chunk_uid": "b", "stage_c_policy_metadata": {"legacy": True}},
            {"chunk_uid": "a", "stage_c_policy_metadata": {"legacy": True}},
        ]
        selected = [{"chunk_uid": "a", "stage_b_policy": {"action": "retain"}}]
        removed = [{"chunk_uid": "b", "stage_b_policy": {"action": "remove"}}]
        universe = module._stage_b_materialization_universe(passed, selected, removed)
        assert [row["chunk_uid"] for row in universe] == ["b", "a"]
        assert [row["stage_b_policy"]["action"] for row in universe] == ["remove", "retain"]
        assert all("stage_c_policy_metadata" not in row for row in universe)

    print("[curation-runtime] generic cross-domain materialization: pass")
    return 0


if __name__ == "__main__":
    test_chunk_text_preserves_code_layout_when_a_paragraph_exceeds_limit()
    test_chunk_text_preserves_an_unbroken_long_line()
    test_runtime_output_preflight_fails_before_expensive_work()
    test_semantic_artifact_preflight_runs_before_quality_scoring()
    test_auto_runtime_artifacts_are_consumed_by_quality_and_coverage()
    raise SystemExit(main())
