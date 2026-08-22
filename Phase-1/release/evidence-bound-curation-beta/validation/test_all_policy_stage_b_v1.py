#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all_policy_stage_b import apply_quality_policy, apply_redundancy_policy
from quality_model_evidence import QualityDecision, QualityPolicyEvidence
from redundancy_equivalence import RedundancyMode
from redundancy_v2 import RedundancySettings


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _result(policy_id: str, decision: QualityDecision) -> QualityPolicyEvidence:
    return QualityPolicyEvidence(
        policy_id=policy_id,
        decision=decision,
        reason_codes=(f"quality_ranker_{decision.value}",),
        class_probabilities=((decision.value, 1.0),),
        failure_probability=1.0 if decision is QualityDecision.FAIL else 0.0,
        failure_threshold=0.7,
        prediction_confidence=1.0,
        minimum_decision_confidence=0.7,
        out_of_distribution=False,
        ranker_artifact_sha256="a" * 64,
    )


def _panel(failed_policy_id: str | None = None) -> tuple[QualityPolicyEvidence, ...]:
    return tuple(
        _result(
            policy_id,
            QualityDecision.FAIL if policy_id == failed_policy_id else QualityDecision.PASS,
        )
        for policy_id in POLICY_IDS
    )


def test_all_stage_b_policies_remove_with_typed_evidence() -> None:
    common = " ".join(f"context{index}" for index in range(110))
    rows = (
        {"chunk_uid": "near-a", "text": f"{common} alpha conclusion", "token_proxy": 112},
        {"chunk_uid": "near-b", "text": f"{common} beta conclusion", "token_proxy": 112},
        {"chunk_uid": "quality-bad", "text": "navigation metadata only", "token_proxy": 3},
        {"chunk_uid": "keep", "text": "A coherent theorem proof with a checked conclusion.", "token_proxy": 8},
    )
    redundancy = apply_redundancy_policy(
        rows,
        mode=RedundancyMode.FRAMEWORK,
        settings=RedundancySettings(
            near_min_tokens=96,
            near_max_changed_ratio=0.01,
            near_max_changed_tokens=2,
        ),
    )

    assert {row["chunk_uid"] for row in redundancy.survivors} == {
        "near-a",
        "quality-bad",
        "keep",
    }
    assert len(redundancy.removals) == 1
    removed = redundancy.removals[0]
    assert removed["chunk_uid"] == "near-b"
    assert removed["stage_b_policy"]["representative_chunk_uid"] == "near-a"
    assert removed["stage_b_redundancy_v2"]["witness_kind"] == "bounded_near_substitute"

    quality_results = {
        str(row["chunk_uid"]): _panel(
            "q3_substantive_payload" if row["chunk_uid"] == "quality-bad" else None
        )
        for row in redundancy.survivors
    }
    quality = apply_quality_policy(
        redundancy.survivors,
        results_by_chunk=quality_results,
    )

    assert {row["chunk_uid"] for row in quality.survivors} == {"near-a", "keep"}
    assert len(quality.not_selected) == 1
    quality_removed = quality.not_selected[0]
    assert quality_removed["chunk_uid"] == "quality-bad"
    assert quality_removed["quality_stage_decision"]["stage_b_action"] == "not_select"
    assert isinstance(quality_removed["quality_stage_decision"]["failed_policy_ids"], list)
    assert isinstance(quality_removed["quality_stage_decision"]["passed_policy_ids"], list)
    assert quality_removed["stage_b_policy"]["removed_reason"] == "quality_qualified_fail"
    assert quality.audit["all_input_chunks_received_quality_decision"] is True
    assert quality.audit["benchmark_outcomes_read"] is False
    assert quality.audit["utility_read"] is False


if __name__ == "__main__":
    test_all_stage_b_policies_remove_with_typed_evidence()
    print("[all-policy-stage-b-v2] redundancy and Quality support gate: pass")
