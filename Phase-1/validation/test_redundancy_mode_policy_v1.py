#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redundancy_mode_policy import (
    DeclaredEquivalenceWitness,
    RedundancyMode,
    WitnessDecision,
    WitnessKind,
    build_redundancy_plan,
    evaluate_equivalence_authority,
)
from redundancy_v2 import RedundancySettings, RedundancyUnit


CONTRACT = ROOT / "configs" / "redundancy_v2.json"


def _long_prose() -> str:
    return " ".join(
        (
            "A reproducible training corpus preserves every observable condition and conclusion",
            *(f"context{index}" for index in range(80)),
        )
    )


def _witness(left: str, right: str) -> DeclaredEquivalenceWitness:
    return DeclaredEquivalenceWitness(
        left_uid=left,
        right_uid=right,
        verifier_id="python-ast-equivalence",
        verifier_version="3.12-v1",
        decision=WitnessDecision.EQUIVALENT,
        artifact_sha256=hashlib.sha256(f"{left}:{right}".encode()).hexdigest(),
    )


def test_normal_and_hard_share_exact_policy_but_use_different_witness_authority() -> None:
    base = "A deterministic record keeps its input hash and final representative linkage for audit."
    longer = f"Introduction. {base} Appendix material remains attached to the same record."
    units = (RedundancyUnit("short", base), RedundancyUnit("long", longer))

    normal = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.NORMAL, exhaustive=True)
    hard = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.HARD, exhaustive=True)

    assert normal.proposed_survivor_uids == ("long", "short")
    assert hard.proposed_survivor_uids == ("long",)
    assert hard.removals[0].removed_uid == "short"
    assert hard.removals[0].representative_uid == "long"
    assert hard.removals[0].witness_kind is WitnessKind.EXACT_TOKEN_CONTAINMENT
    assert hard.removals[0].coverage_veto_required is True


def test_hard_accepts_token_preserving_prose_reflow_but_normal_retains_it() -> None:
    left_text = _long_prose()
    right_text = left_text.replace(" context20 ", "\ncontext20 ", 1)
    units = (RedundancyUnit("a", left_text), RedundancyUnit("b", right_text))

    authority = evaluate_equivalence_authority(units[0], units[1], RedundancySettings())
    normal = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.NORMAL, exhaustive=True)
    hard = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.HARD, exhaustive=True)

    assert authority.witness is not None
    assert authority.witness.kind is WitnessKind.TOKEN_PRESERVING_PROSE_REFLOW
    assert authority.normal_authority is False
    assert authority.hard_authority is True
    assert len(normal.proposed_survivor_uids) == 2
    assert hard.proposed_survivor_uids == ("a",)


def test_declared_verifier_can_authorize_hard_near_equivalence_only() -> None:
    padding = "\n".join(f"# invariant context {index}" for index in range(40))
    left = RedundancyUnit("code-a", f"def stable():\n    return 1\n{padding}")
    right = RedundancyUnit("code-b", f"def stable():\n    return 1;\n{padding}")
    declared = _witness(left.uid, right.uid)

    normal = build_redundancy_plan(
        (left, right), RedundancySettings(), RedundancyMode.NORMAL, (declared,), exhaustive=True
    )
    hard = build_redundancy_plan(
        (left, right), RedundancySettings(), RedundancyMode.HARD, (declared,), exhaustive=True
    )

    assert len(normal.proposed_survivor_uids) == 2
    assert hard.proposed_survivor_uids == ("code-a",)
    assert hard.removals[0].witness_kind is WitnessKind.DECLARED_EQUIVALENCE_VERIFIER
    authority = next(
        decision
        for decision in hard.authority_decisions
        if decision.witness is not None
        and decision.witness.kind is WitnessKind.DECLARED_EQUIVALENCE_VERIFIER
    )
    assert authority.witness is not None
    assert authority.witness.verifier_id == declared.verifier_id
    assert authority.witness.verifier_version == declared.verifier_version
    assert authority.witness.source_artifact_sha256 == declared.artifact_sha256


def test_substantive_changes_and_unproved_similarity_never_receive_authority() -> None:
    long = _long_prose()
    numeric_left = RedundancyUnit("numeric-a", f"{long} The frozen value is 200.")
    numeric_right = RedundancyUnit("numeric-b", f"{long} The frozen value is 404.")
    lexical_left = RedundancyUnit("lexical-a", long.replace("conclusion", "inspection"))
    lexical_right = RedundancyUnit("lexical-b", long.replace("conclusion", "review"))

    numeric = evaluate_equivalence_authority(
        numeric_left, numeric_right, RedundancySettings(), _witness(numeric_left.uid, numeric_right.uid)
    )
    lexical = evaluate_equivalence_authority(lexical_left, lexical_right, RedundancySettings())

    assert "numeric_constant_changed" in numeric.relation.evidence.substantive_difference_codes
    assert numeric.hard_authority is False
    assert lexical.hard_authority is False
    assert numeric.witness is None
    assert lexical.witness is None


def test_repeated_span_and_semantic_candidates_do_not_delete_whole_records() -> None:
    repeated = (
        "Every artifact records the policy identifier, input hash, output hash, reason code, "
        "and final representative linkage for independent verification."
    )
    units = (
        RedundancyUnit("alpha", f"Alpha overview.\n\n{repeated}\n\nAlpha command line details."),
        RedundancyUnit("beta", f"Beta overview.\n\n{repeated}\n\nBeta Python details."),
    )
    plan = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.HARD, exhaustive=True)

    assert plan.proposed_survivor_uids == ("alpha", "beta")
    assert plan.removals == ()


def test_hard_survivors_are_always_a_subset_of_normal_survivors() -> None:
    exact = "identical payload with a stable record identity"
    prose = _long_prose()
    units = (
        RedundancyUnit("exact-a", exact),
        RedundancyUnit("exact-b", exact),
        RedundancyUnit("reflow-a", prose),
        RedundancyUnit("reflow-b", prose.replace(" context30 ", "\ncontext30 ", 1)),
    )
    normal = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.NORMAL, exhaustive=True)
    hard = build_redundancy_plan(units, RedundancySettings(), RedundancyMode.HARD, exhaustive=True)

    assert set(hard.proposed_survivor_uids) <= set(normal.proposed_survivor_uids)
    assert all(removal.benchmark_outcomes_read is False for removal in hard.removals)
    assert all(removal.utility_read is False for removal in hard.removals)


def test_transitive_family_uses_one_stable_id_and_representative_trace() -> None:
    payload = "same payload repeated as one audited equivalence family"
    units = tuple(
        RedundancyUnit(uid, payload)
        for uid in ("family-c", "family-a", "family-b")
    )

    plan = build_redundancy_plan(
        units, RedundancySettings(), RedundancyMode.NORMAL, exhaustive=True
    )

    assert plan.proposed_survivor_uids == ("family-a",)
    assert len(plan.families) == 1
    assert plan.families[0].representative_uid == "family-a"
    assert plan.families[0].member_uids == ("family-a", "family-b", "family-c")
    assert {removal.family_id for removal in plan.removals} == {
        plan.families[0].family_id
    }
    assert {removal.evidence_sha256 for removal in plan.removals} == {
        plan.families[0].evidence_sha256
    }


def test_mode_contract_is_candidate_only_and_has_no_external_selector_inputs() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["runtime_activation"] is False
    assert contract["mode_policy_status"] == "implementation_candidate_validation_pending"
    assert contract["operating_points"]["normal"]["uncertain_action"] == "retain"
    assert contract["operating_points"]["hard"]["coverage_veto_required"] is True
    assert contract["declared_equivalence_verifier_contract"]["benchmark_outcomes_read"] is False
    assert contract["declared_equivalence_verifier_contract"]["utility_read"] is False
    assert "benchmark_outcomes" in contract["forbidden_inputs"]
    assert "target_retention_fraction" in contract["forbidden_inputs"]


if __name__ == "__main__":
    test_normal_and_hard_share_exact_policy_but_use_different_witness_authority()
    test_hard_accepts_token_preserving_prose_reflow_but_normal_retains_it()
    test_declared_verifier_can_authorize_hard_near_equivalence_only()
    test_substantive_changes_and_unproved_similarity_never_receive_authority()
    test_repeated_span_and_semantic_candidates_do_not_delete_whole_records()
    test_hard_survivors_are_always_a_subset_of_normal_survivors()
    test_transitive_family_uses_one_stable_id_and_representative_trace()
    test_mode_contract_is_candidate_only_and_has_no_external_selector_inputs()
    print("[redundancy-mode-policy-v1] witness authority and mode containment: pass")
