#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_fallback_evidence import (
    MissingQualityFallbackEvidenceError,
    load_quality_fallback_evidence,
    write_quality_fallback_requests,
    write_quality_local_evidence,
)
from quality_model_evidence import (
    QualityDecision,
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
)
from quality_operating_points import QualityAction, decide_quality_action
from quality_teacher_observation_codec import quality_runtime_sha256, quality_task_id


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _local(
    policy_id: str,
    decision: QualityDecision,
    *,
    qualified_fail: bool = False,
) -> QualityPolicyEvidence:
    return QualityPolicyEvidence(
        policy_id=policy_id,
        decision=decision,
        reason_codes=(f"local_{decision.value}",),
        class_probabilities=((decision.value, 1.0),),
        failure_probability=0.95 if qualified_fail else 0.0,
        failure_threshold=0.8,
        prediction_confidence=0.95,
        minimum_decision_confidence=0.8,
        out_of_distribution=False,
        ranker_artifact_sha256="a" * 64,
    )


def _local_panel(*passed: str) -> tuple[QualityPolicyEvidence, ...]:
    return tuple(
        _local(
            policy_id,
            QualityDecision.PASS if policy_id in passed else QualityDecision.ABSTAIN,
        )
        for policy_id in POLICY_IDS
    )


def _teacher_panel(*, passed: tuple[str, ...] = (), failed: tuple[str, ...] = ()) -> tuple[TeacherQualityPolicyEvidence, ...]:
    return tuple(
        TeacherQualityPolicyEvidence(
            policy_id=policy_id,
            decision=(
                QualityDecision.FAIL
                if policy_id in failed
                else QualityDecision.PASS
                if policy_id in passed
                else QualityDecision.ABSTAIN
            ),
            reason_codes=("teacher_evidence",),
            observation_sha256="b" * 64,
        )
        for policy_id in POLICY_IDS
    )


def test_local_substantive_payload_alone_requires_fallback() -> None:
    try:
        decide_quality_action(
            _local_panel("q3_substantive_payload"),
            coverage_veto=False,
        )
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Q3 alone must not authorize local retention")


def test_local_coherence_and_relations_without_payload_require_fallback() -> None:
    try:
        decide_quality_action(
            _local_panel("q2_semantic_coherence", "q4_learnable_relations"),
            coverage_veto=False,
        )
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Q2 and Q4 without Q3 must not authorize retention")


def test_local_coherence_payload_and_relations_are_positive_support() -> None:
    decision = decide_quality_action(
        _local_panel(
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        ),
        coverage_veto=False,
    )

    assert decision.action is QualityAction.RETAIN
    assert decision.reason_code == "quality_local_positive_support"


def test_unsupported_local_result_requires_complete_teacher_evidence() -> None:
    try:
        decide_quality_action(_local_panel(), coverage_veto=False)
    except MissingQualityFallbackEvidenceError as error:
        assert error.policy_ids == POLICY_IDS
    else:
        raise AssertionError("Unsupported local evidence must not silently retain a chunk")


def test_teacher_confirmed_fail_excludes_the_chunk() -> None:
    decision = decide_quality_action(
        _local_panel(),
        coverage_veto=False,
        teacher_results=_teacher_panel(failed=("q3_substantive_payload",)),
    )

    assert decision.action is QualityAction.NOT_SELECT
    assert decision.reason_code == "quality_teacher_confirmed_fail"
    assert decision.decision_source == "luna_fallback"


def test_teacher_positive_support_retains_the_chunk() -> None:
    decision = decide_quality_action(
        _local_panel(),
        coverage_veto=False,
        teacher_results=_teacher_panel(
            passed=(
                "q2_semantic_coherence",
                "q3_substantive_payload",
                "q4_learnable_relations",
            )
        ),
    )

    assert decision.action is QualityAction.RETAIN
    assert decision.reason_code == "quality_teacher_positive_support"


def test_teacher_substantive_payload_alone_is_not_positive_support() -> None:
    decision = decide_quality_action(
        _local_panel(),
        coverage_veto=False,
        teacher_results=_teacher_panel(passed=("q3_substantive_payload",)),
    )

    assert decision.action is QualityAction.NOT_SELECT
    assert decision.reason_code == "quality_teacher_no_positive_support"


def test_teacher_without_positive_support_excludes_instead_of_abstain_retain() -> None:
    decision = decide_quality_action(
        _local_panel(),
        coverage_veto=False,
        teacher_results=_teacher_panel(),
    )

    assert decision.action is QualityAction.NOT_SELECT
    assert decision.reason_code == "quality_teacher_no_positive_support"


def test_fallback_loader_rejects_text_identity_mismatch() -> None:
    text = "def add(a, b):\n    return a + b"
    panel_sha256 = "c" * 64
    runtime_sha256 = quality_runtime_sha256()
    stale_text_sha256 = "0" * 64
    observation = {
        "schema_version": "quality-teacher-corpus-observation-v3",
        "aggregation_strategy": "single_teacher_confirmed_fail",
        "teacher_panel_sha256": panel_sha256,
        "quality_runtime_sha256": runtime_sha256,
        "chunk_uid": "chunk-1",
        "text_sha256": stale_text_sha256,
        "task_id": quality_task_id(
            panel_sha256,
            runtime_sha256,
            "chunk-1",
            stale_text_sha256,
        ),
        "policy_results": [
            {
                "policy_id": policy_id,
                "panel_decision": "pass",
                "decision_reason_codes": ["teacher_evidence"],
            }
            for policy_id in POLICY_IDS
        ],
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "observations.jsonl"
        path.write_text(json.dumps(observation) + "\n", encoding="utf-8")
        try:
            load_quality_fallback_evidence(
                path,
                {"chunk-1": text},
                expected_panel_sha256=panel_sha256,
            )
        except RuntimeError as error:
            assert "text identity mismatch" in str(error)
        else:
            raise AssertionError("Stale teacher evidence must be rejected")


def test_request_writer_emits_only_chunks_without_local_support() -> None:
    rows = (
        {"chunk_uid": "supported", "text": "substantive payload"},
        {"chunk_uid": "unsupported", "text": "fragment"},
    )
    local = {
        "supported": _local_panel(
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        ),
        "unsupported": _local_panel(),
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        audit = write_quality_fallback_requests(path, rows, local)
        written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["chunk_uid"] for row in written] == ["unsupported"]
    assert written[0]["text_sha256"] == hashlib.sha256(b"fragment").hexdigest()
    assert audit["request_chunks"] == 1


def test_local_evidence_writer_records_all_three_pre_fallback_states() -> None:
    rows = (
        {"chunk_uid": "supported", "text": "substantive payload"},
        {"chunk_uid": "failed", "text": "broken payload"},
        {"chunk_uid": "unsupported", "text": "fragment"},
    )
    local = {
        "supported": _local_panel(
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        ),
        "failed": tuple(
            _local(
                policy_id,
                QualityDecision.FAIL
                if policy_id == "q3_substantive_payload"
                else QualityDecision.ABSTAIN,
                qualified_fail=policy_id == "q3_substantive_payload",
            )
            for policy_id in POLICY_IDS
        ),
        "unsupported": _local_panel(),
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "local.jsonl"
        audit = write_quality_local_evidence(path, rows, local)
        written = {
            row["chunk_uid"]: row
            for row in (
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            )
        }

    assert written["supported"]["local_action"] == "retain"
    assert written["failed"]["local_action"] == "not_select"
    assert written["unsupported"]["local_action"] == "luna_fallback_required"
    assert audit["decision_counts"] == {
        "retain": 1,
        "not_select": 1,
        "luna_fallback_required": 1,
    }


if __name__ == "__main__":
    test_local_substantive_payload_alone_requires_fallback()
    test_local_coherence_and_relations_without_payload_require_fallback()
    test_local_coherence_payload_and_relations_are_positive_support()
    test_unsupported_local_result_requires_complete_teacher_evidence()
    test_teacher_confirmed_fail_excludes_the_chunk()
    test_teacher_positive_support_retains_the_chunk()
    test_teacher_substantive_payload_alone_is_not_positive_support()
    test_teacher_without_positive_support_excludes_instead_of_abstain_retain()
    test_fallback_loader_rejects_text_identity_mismatch()
    test_request_writer_emits_only_chunks_without_local_support()
    test_local_evidence_writer_records_all_three_pre_fallback_states()
    print("[quality-fallback-authority-v1] positive support and Luna fallback: pass")
