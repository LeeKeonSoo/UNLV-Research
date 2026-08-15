#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_batch_runtime import (
    PolicySetBatchGenerationRequest,
    build_policy_set_response_schema,
)
from quality_teacher_luna_batch_runner import MODEL_ID, TEACHER_ID, _unit, alias_units, parser
from quality_teacher_panel import load_teacher_panel
from quality_teacher_runtime import EvaluationUnit


def test_alias_units_replaces_long_provider_ids_and_preserves_local_linkage() -> None:
    units = tuple(
        EvaluationUnit(
            unit_id=f"source::very-long-content-digest-{index:02d}",
            text=f"payload {index}",
            declared_context=None,
            attached_evidence=(),
        )
        for index in range(3)
    )

    aliased, linkage = alias_units(units)

    assert [unit.unit_id for unit in aliased] == ["u00", "u01", "u02"]
    assert linkage == [
        {"alias": "u00", "unit_id": "source::very-long-content-digest-00"},
        {"alias": "u01", "unit_id": "source::very-long-content-digest-01"},
        {"alias": "u02", "unit_id": "source::very-long-content-digest-02"},
    ]
    assert [unit.text for unit in aliased] == [unit.text for unit in units]


def test_unit_restores_declared_verifier_from_fixture_row() -> None:
    unit = _unit(
        {
            "chunk_uid": "q1-fixture-001",
            "text": "assert 2 + 2 == 5",
            "quality_declared_context": "controlled arithmetic fixture",
            "quality_attached_evidence": ["correct_sum=4"],
            "quality_declared_verifier": {
                "verifier_id": "controlled-local-verifier-v1",
                "status": "fail",
                "evidence_sha256": "a" * 64,
            },
        }
    )

    assert unit.declared_verifier is not None
    assert unit.declared_verifier.status == "fail"
    assert unit.declared_verifier.evidence_sha256 == "a" * 64


def test_parser_exposes_explicit_batch_cancel_command() -> None:
    args = parser().parse_args(["cancel", "--output-dir", "batch-artifact"])

    assert args.command == "cancel"
    assert args.handler.__name__ == "cancel"


def test_strict_schema_binds_reason_codes_to_policy_and_decision() -> None:
    panel = load_teacher_panel(ROOT / "configs" / "quality_teacher_luna_single_v1.json")
    request = PolicySetBatchGenerationRequest(
        teacher_id=TEACHER_ID,
        model_id=MODEL_ID,
        policies=panel.policies,
        units=(
            EvaluationUnit(
                unit_id="u14",
                text="Site | Accept cookies | Privacy settings | Sign in",
                declared_context="English general prose fixture.",
                attached_evidence=("boilerplate_only",),
            ),
        ),
        pass_index=2,
        blind_run_id="blind-run",
        schema_retry=True,
    )

    schema = build_policy_set_response_schema(request.policies, request.units)
    policy_variants = schema["properties"]["units"]["items"]["properties"][
        "policies"
    ]["items"]["anyOf"]
    q4_fail = next(
        variant
        for variant in policy_variants
        if variant["properties"]["policy_id"]["enum"]
        == ["q4_learnable_relations"]
        and variant["properties"]["decision"]["enum"] == ["fail"]
    )

    allowed = q4_fail["properties"]["reason_codes"]["items"]["enum"]
    assert "navigation_only" not in allowed
    assert allowed == [
        "unconnected_token_set",
        "label_only_without_relation",
        "fragment_set_without_relation",
    ]


if __name__ == "__main__":
    test_alias_units_replaces_long_provider_ids_and_preserves_local_linkage()
    test_unit_restores_declared_verifier_from_fixture_row()
    test_parser_exposes_explicit_batch_cancel_command()
    test_strict_schema_binds_reason_codes_to_policy_and_decision()
    print("[quality-teacher-luna-batch-runner-v1] alias linkage contract: pass")
