#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redundancy_v2 import (
    RedundancySettings,
    RedundancyUnit,
    build_redundancy_graph,
    classify_relation,
)
from redundancy_v2_audit import RedundancyAuditCase, build_redundancy_audit
from redundancy_v2_retrieval import retrieve_candidate_pairs


FIXTURES = ROOT / "validation" / "fixtures" / "redundancy_v2_cases.json"
CONTRACT = ROOT / "configs" / "redundancy_v2.json"


def _cases() -> list[dict[str, str | bool]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return payload["cases"]


def _audit_cases() -> tuple[RedundancyAuditCase, ...]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    cases = [
        RedundancyAuditCase(
            case_id=str(case["id"]),
            role=str(case["role"]),
            left=RedundancyUnit(f"{case['id']}:left", str(case["left"])),
            right=RedundancyUnit(f"{case['id']}:right", str(case["right"])),
            expected_relation=str(case["expected_relation"]),
            semantic_candidate=bool(case.get("semantic_candidate", False)),
        )
        for case in payload["cases"]
    ]
    for index, text in enumerate(payload["metamorphic_equivalence_payloads"]):
        formatting_variant = text.replace("\n", "\r\n") if "\n" in text else text + "\r\n"
        cases.append(
            RedundancyAuditCase(
                case_id=f"metamorphic_exact_{index}",
                role="safe_family_positive",
                left=RedundancyUnit(f"metamorphic_exact_{index}:left", text),
                right=RedundancyUnit(f"metamorphic_exact_{index}:right", text),
                expected_relation="exact_equivalent",
            )
        )
        cases.append(
            RedundancyAuditCase(
                case_id=f"metamorphic_formatting_{index}",
                role="safe_family_positive",
                left=RedundancyUnit(f"metamorphic_formatting_{index}:left", text),
                right=RedundancyUnit(f"metamorphic_formatting_{index}:right", formatting_variant),
                expected_relation="formatting_equivalent",
            )
        )
    return tuple(cases)


def test_typed_relations_preserve_substantive_differences() -> None:
    settings = RedundancySettings()
    for case in _cases():
        relation = classify_relation(
            RedundancyUnit("left", str(case["left"])),
            RedundancyUnit("right", str(case["right"])),
            settings,
            semantic_candidate=bool(case.get("semantic_candidate", False)),
        )

        assert relation.relation.value == case["expected_relation"], case["id"]
        assert relation.selection_authority is False
        assert relation.benchmark_outcomes_read is False
        assert relation.utility_read is False
        if case["role"] == "retain_control":
            assert relation.safe_family_edge is False, case["id"]


def test_family_graph_uses_only_equivalence_edges_and_defers_representative() -> None:
    units = (
        RedundancyUnit("a", "same payload with stable tokens"),
        RedundancyUnit("b", "same payload with stable tokens\r\n"),
        RedundancyUnit("c", "same payload with stable tokens"),
        RedundancyUnit("d", "same payload with changed tokens"),
    )
    graph = build_redundancy_graph(units, RedundancySettings(), exhaustive=True)

    family = next(family for family in graph.families if "a" in family.member_uids)
    assert family.member_uids == ("a", "b", "c")
    assert family.final_representative_uid is None
    assert family.representative_selection_deferred is True
    assert "d" not in family.member_uids
    assert all(edge.relation.value in {"exact_equivalent", "formatting_equivalent"} for edge in family.edges)


def test_retrieval_recovers_equivalence_near_containment_and_span_candidates() -> None:
    selected = {
        "exact_multilingual",
        "formatting_general",
        "long_non_substantive_substitution",
        "contained_payload",
        "repeated_span",
    }
    units: list[RedundancyUnit] = []
    expected_pairs: set[tuple[str, str]] = set()
    for case in _cases():
        case_id = str(case["id"])
        if case_id not in selected:
            continue
        left_uid = f"{case_id}:left"
        right_uid = f"{case_id}:right"
        units.extend((RedundancyUnit(left_uid, str(case["left"])), RedundancyUnit(right_uid, str(case["right"]))))
        expected_pairs.add((left_uid, right_uid))

    retrieved = retrieve_candidate_pairs(
        tuple(units),
        RedundancySettings(retrieve_repeated_span_candidates=True),
    )
    observed = {(pair.left_uid, pair.right_uid) for pair in retrieved}

    assert expected_pairs <= observed


def test_runtime_retrieval_does_not_expand_candidate_only_repeated_spans() -> None:
    repeated = "one two three four five six seven eight nine ten eleven twelve"
    units = (
        RedundancyUnit("a", f"alpha\n\n{repeated}"),
        RedundancyUnit("b", f"beta\n\n{repeated}"),
        RedundancyUnit("c", f"gamma\n\n{repeated}"),
    )

    runtime_pairs = retrieve_candidate_pairs(
        units,
        RedundancySettings(containment_min_tokens=24),
    )
    candidate_pairs = retrieve_candidate_pairs(
        units,
        RedundancySettings(
            containment_min_tokens=24,
            retrieve_repeated_span_candidates=True,
        ),
    )

    assert runtime_pairs == ()
    assert len(candidate_pairs) == 3
    assert all(
        pair.retrieval_reasons == ("repeated_paragraph_digest",)
        for pair in candidate_pairs
    )


def test_containment_retrieval_requires_both_payload_anchors() -> None:
    shared = " ".join(f"shared{index}" for index in range(12))
    units = tuple(
        RedundancyUnit(
            uid,
            f"{shared} " + " ".join(f"{uid}_unique{index}" for index in range(20)),
        )
        for uid in ("a", "b", "c")
    )

    retrieved = retrieve_candidate_pairs(
        units,
        RedundancySettings(
            containment_min_tokens=12,
            retrieval_min_tokens=1_000,
        ),
    )

    assert retrieved == ()


def test_fixture_audit_has_zero_safe_family_false_positives() -> None:
    report = build_redundancy_audit(_audit_cases(), RedundancySettings(), confidence_level=0.95)

    assert report.safe_family_positive_count >= 20
    assert report.safe_family_false_negative_count == 0
    assert report.safe_family_false_negative_upper_bound <= 0.15
    assert report.retain_control_count >= 20
    assert report.retain_safe_family_false_positive_count == 0
    assert report.retain_safe_family_false_positive_upper_bound <= 0.15
    assert report.relation_mismatch_count == 0
    assert report.passed is True


def test_contract_activates_redundancy_v2_with_coverage_veto() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "all_policy_runtime_experiment_v1"
    assert contract["runtime_activation"] is True
    assert contract["mode_policy_status"] == (
        "frozen_runtime_experiment_pending_external_validation"
    )
    assert contract["runtime_authority"] == (
        "stage_b_policy_may_propose_removal_subject_to_stage_c_coverage_veto"
    )
    assert contract["development_ablation_ready"] is True
    assert contract["development_gate_registry_sha256"] == "eb27475d77414f36173448d44c9e53871e5c22195d265890be024f7af43b1c41"
    assert contract["development_gate_report_sha256"] == "5d9d99d117f4f88bb1fca2e34f52d3444b53c52e9db8e2f979895d1700f00e51"
    assert contract["safe_family_relations"] == ["exact_equivalent", "formatting_equivalent"]
    assert contract["representative_selection"] == (
        "stage_b_proposes_payload_superset_then_stable_uid_and_stage_c_coverage_finalizes"
    )
    assert contract["single_global_similarity_threshold"] is False


if __name__ == "__main__":
    test_typed_relations_preserve_substantive_differences()
    test_family_graph_uses_only_equivalence_edges_and_defers_representative()
    test_retrieval_recovers_equivalence_near_containment_and_span_candidates()
    test_runtime_retrieval_does_not_expand_candidate_only_repeated_spans()
    test_containment_retrieval_requires_both_payload_anchors()
    test_fixture_audit_has_zero_safe_family_false_positives()
    test_contract_activates_redundancy_v2_with_coverage_veto()
    print("[redundancy-v2] typed relation, retrieval, and family contract: pass")
