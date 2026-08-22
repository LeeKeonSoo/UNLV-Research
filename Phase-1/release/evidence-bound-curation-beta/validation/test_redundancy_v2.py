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
    classify_relation,
)
from redundancy_v2_retrieval import retrieve_candidate_pairs


FIXTURES = ROOT / "validation" / "fixtures" / "redundancy_v2_cases.json"
CONTRACT = ROOT / "configs" / "redundancy_v2.json"


def _cases() -> list[dict[str, str | bool]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return payload["cases"]


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


def test_exact_and_line_ending_equivalence_are_metamorphically_stable() -> None:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    settings = RedundancySettings()
    for index, text in enumerate(payload["metamorphic_equivalence_payloads"]):
        exact = classify_relation(
            RedundancyUnit(f"exact-{index}-left", text),
            RedundancyUnit(f"exact-{index}-right", text),
            settings,
        )
        formatting = classify_relation(
            RedundancyUnit(f"format-{index}-left", text),
            RedundancyUnit(
                f"format-{index}-right",
                text.replace("\n", "\r\n") if "\n" in text else text + "\r\n",
            ),
            settings,
        )
        assert exact.relation.value == "exact_equivalent"
        assert formatting.relation.value == "formatting_equivalent"
        assert exact.safe_family_edge is True
        assert formatting.safe_family_edge is True


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
    assert len(candidate_pairs) == 2
    assert {pair.left_uid for pair in candidate_pairs} == {"a"}
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


def test_character_minhash_retrieves_reformatted_candidate_without_authorizing_it() -> None:
    payload = " ".join(f"evidence{index}" for index in range(90))
    units = (
        RedundancyUnit("plain", payload),
        RedundancyUnit("reformatted", payload.replace(" ", "\n", 12)),
    )

    retrieved = retrieve_candidate_pairs(units, RedundancySettings())
    pair = next(item for item in retrieved if item.left_uid == "plain")

    assert "character_minhash_lsh" in pair.retrieval_reasons


def test_semantic_similarity_is_not_a_runtime_redundancy_authority() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["semantic_similarity_action"] == "coverage_audit_only"
    assert contract["retrieval"]["character_ngram_size"] == 24
    assert contract["retrieval"]["character_minhash_candidate_only"] is True
    assert contract["retrieval"]["character_minhash_variant"] == (
        "densified_one_permutation"
    )
    assert contract["retrieval"]["lsh_bucket_pairing"] == "stable_anchor_star"
    assert contract["complexity_contract"]["all_pairs_similarity_scan"] is False


def test_large_retrieval_buckets_use_linear_star_candidates() -> None:
    payload = " ".join(f"stable{index}" for index in range(90))
    units = tuple(RedundancyUnit(f"unit-{index:03d}", payload) for index in range(100))

    retrieved = retrieve_candidate_pairs(units, RedundancySettings())

    assert len(retrieved) == 99
    assert {pair.left_uid for pair in retrieved} == {"unit-000"}


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
    test_exact_and_line_ending_equivalence_are_metamorphically_stable()
    test_retrieval_recovers_equivalence_near_containment_and_span_candidates()
    test_runtime_retrieval_does_not_expand_candidate_only_repeated_spans()
    test_containment_retrieval_requires_both_payload_anchors()
    test_character_minhash_retrieves_reformatted_candidate_without_authorizing_it()
    test_semantic_similarity_is_not_a_runtime_redundancy_authority()
    test_large_retrieval_buckets_use_linear_star_candidates()
    test_contract_activates_redundancy_v2_with_coverage_veto()
    print("[redundancy-v2] typed relation, retrieval, and family contract: pass")
