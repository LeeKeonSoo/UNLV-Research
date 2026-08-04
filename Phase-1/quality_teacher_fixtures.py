from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from quality_teacher_runtime import DeclaredVerifierEvidence, EvaluationUnit


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)
ROUTES = (
    "code_artifact",
    "mathematical_content",
    "general_prose",
    "table_structured_data",
)


class FixtureClass(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"
    PROTECTED_PASS = "protected_pass"


@dataclass(frozen=True, slots=True)
class BehaviorFixture:
    fixture_id: str
    policy_id: str
    route: str
    fixture_class: FixtureClass
    expected_decision: Literal["pass", "fail", "abstain"]
    expected_reason_code: str
    label_provenance: Literal["deterministic_construction"]
    unit: EvaluationUnit


@dataclass(frozen=True, slots=True)
class ProtectedFixture:
    fixture_id: str
    route: str
    expected_quality_gate: Literal["pass"]
    verifier_evidence: tuple[str, ...]
    unit: EvaluationUnit


_PASS_REASONS = {
    "q1_correctness_evidence": "observable_correctness_evidence",
    "q2_semantic_coherence": "recoverable_semantic_unit",
    "q3_substantive_payload": "substantive_payload_present",
    "q4_learnable_relations": "recoverable_relation_present",
}
_FAIL_REASONS = {
    "q1_correctness_evidence": "locally_checkable_incorrect_result",
    "q2_semantic_coherence": "internal_semantic_contradiction",
    "q3_substantive_payload": "boilerplate_only",
    "q4_learnable_relations": "fragment_set_without_relation",
}
_ABSTAIN_REASONS = {
    "q1_correctness_evidence": "external_knowledge_required",
    "q2_semantic_coherence": "missing_context_may_repair_coherence",
    "q3_substantive_payload": "specialized_payload_uncertain",
    "q4_learnable_relations": "specialized_notation_relation_uncertain",
}


def _pass_payload(route: str, index: int) -> tuple[str, tuple[str, ...]]:
    a = index + 2
    b = index + 3
    total = a + b
    if route == "code_artifact":
        return (
            f"def add_{index}(left, right):\n    return left + right\nassert add_{index}({a}, {b}) == {total}",
            (f"{a} + {b} = {total}",),
        )
    if route == "mathematical_content":
        return (
            f"Let x = {a} and y = {b}. Since addition is closed, x + y = {a} + {b} = {total}.",
            (f"integer_addition:{a}+{b}={total}",),
        )
    if route == "general_prose":
        return (
            f"Trial {index} used {a} blue cards and {b} green cards. Combining the two groups produced {total} cards, so the recorded total follows from the stated counts.",
            (f"stated_counts_sum_to_{total}",),
        )
    return (
        f"category,count\nblue,{a}\ngreen,{b}\ntotal,{total}",
        (f"column_sum:{a}+{b}={total}",),
    )


def _fail_payload(policy_id: str, route: str, index: int) -> tuple[str, tuple[str, ...]]:
    if policy_id == "q1_correctness_evidence":
        text, _ = _pass_payload(route, index)
        expected = (index + 2) + (index + 3)
        return text.replace(str(expected), str(expected + 1)), (f"correct_sum={expected}",)
    if policy_id == "q2_semantic_coherence":
        payloads = {
            "code_artifact": "def area(width, height): return width * height\nThe function always divides width by height.",
            "mathematical_content": "Assume n is even. Therefore n is odd under the same unchanged assumption.",
            "general_prose": "The valve remained closed throughout the trial. The same valve was open throughout the trial.",
            "table_structured_data": "state,value\nclosed,1\nclosed,not-closed",
        }
        return payloads[route], ("same_context_contains_incompatible_claims",)
    if policy_id == "q3_substantive_payload":
        payloads = {
            "code_artifact": "# Copyright 2026 Example\n# All rights reserved.\n# End of header.",
            "mathematical_content": "Home | Mathematics | Previous | Next | Search",
            "general_prose": "Accept cookies | Privacy settings | Sign in | Subscribe",
            "table_structured_data": "title,updated,owner\nIndex page,today,site administrator",
        }
        return payloads[route], ("no_residual_payload_after_boilerplate",)
    payloads = {
        "code_artifact": "alpha beta gamma delta",
        "mathematical_content": "x y z theorem lemma",
        "general_prose": "river copper quiet seventeen",
        "table_structured_data": "label\nalpha\nbeta\ngamma",
    }
    return payloads[route], ("no_entity_operation_or_outcome_relation",)


def _abstain_payload(route: str, index: int) -> tuple[str, tuple[str, ...]]:
    payloads = {
        "code_artifact": f"opaque_intrinsic_{index}(state)",
        "mathematical_content": f"Q_{index} star OMEGA entails bracket_alpha",
        "general_prose": f"The {index}th boundary follows by the customary local convention.",
        "table_structured_data": f"symbol,value\nOMEGA_{index},star",
    }
    return payloads[route], ("specialized_context_not_supplied",)


def _behavior_payload(
    policy_id: str,
    route: str,
    fixture_class: FixtureClass,
    index: int,
) -> tuple[str, tuple[str, ...]]:
    if fixture_class in {FixtureClass.PASS, FixtureClass.PROTECTED_PASS}:
        return _pass_payload(route, index + (100 if fixture_class is FixtureClass.PROTECTED_PASS else 0))
    if fixture_class is FixtureClass.FAIL:
        return _fail_payload(policy_id, route, index)
    return _abstain_payload(route, index)


def _verifier(
    fixture_id: str,
    status: Literal["pass", "fail"],
    evidence: tuple[str, ...],
) -> DeclaredVerifierEvidence:
    payload = "\n".join((fixture_id, status, *evidence)).encode("utf-8")
    return DeclaredVerifierEvidence(
        verifier_id="controlled-local-verifier-v1",
        status=status,
        evidence_sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_behavior_fixture_matrix(samples_per_cell: int = 8) -> tuple[BehaviorFixture, ...]:
    fixtures: list[BehaviorFixture] = []
    for policy_id in POLICY_IDS:
        for route in ROUTES:
            for fixture_class in FixtureClass:
                for index in range(samples_per_cell):
                    fixture_id = f"{policy_id}-{route}-{fixture_class.value}-{index:03d}"
                    text, evidence = _behavior_payload(policy_id, route, fixture_class, index)
                    expected = "pass" if fixture_class is FixtureClass.PROTECTED_PASS else fixture_class.value
                    verifier = (
                        _verifier(fixture_id, expected, evidence)
                        if policy_id == "q1_correctness_evidence" and expected in {"pass", "fail"}
                        else None
                    )
                    reason_map = (
                        _PASS_REASONS
                        if expected == "pass"
                        else _FAIL_REASONS if expected == "fail" else _ABSTAIN_REASONS
                    )
                    fixtures.append(
                        BehaviorFixture(
                            fixture_id=fixture_id,
                            policy_id=policy_id,
                            route=route,
                            fixture_class=fixture_class,
                            expected_decision=expected,
                            expected_reason_code=reason_map[policy_id],
                            label_provenance="deterministic_construction",
                            unit=EvaluationUnit(
                                unit_id=fixture_id,
                                text=text,
                                declared_context=f"English {route} controlled fixture.",
                                attached_evidence=evidence,
                                declared_verifier=verifier,
                            ),
                        )
                    )
    return tuple(fixtures)


def build_protected_fixture_set(samples_per_route: int = 200) -> tuple[ProtectedFixture, ...]:
    fixtures: list[ProtectedFixture] = []
    for route in ROUTES:
        for index in range(samples_per_route):
            fixture_id = f"protected-{route}-{index:04d}"
            text, evidence = _pass_payload(route, index + 1000)
            fixtures.append(
                ProtectedFixture(
                    fixture_id=fixture_id,
                    route=route,
                    expected_quality_gate="pass",
                    verifier_evidence=evidence,
                    unit=EvaluationUnit(
                        unit_id=fixture_id,
                        text=text,
                        declared_context=f"English {route} protected controlled fixture.",
                        attached_evidence=evidence,
                        declared_verifier=_verifier(fixture_id, "pass", evidence),
                    ),
                )
            )
    return tuple(fixtures)
