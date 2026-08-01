#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs" / "general_provider_candidate_decision_v2.json"


def test_rejected_components_cannot_be_ensembled_into_runtime_authority() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    assert decision["decision"] == "reject_all_evaluated_general_provider_candidates"
    assert decision["runtime_activation"] is False
    assert decision["ensemble_activation"] is False
    assert all(candidate["decision"] == "reject" for candidate in decision["candidates"])
    assert all("source_transfer" in candidate["blocking_gates"] for candidate in decision["candidates"])
    assert all("semantic_destruction" in candidate["blocking_gates"] for candidate in decision["candidates"])


def test_decision_preserves_component_scope() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    assert all(candidate["supported_heads"] == ["route_specific_evidence"] for candidate in decision["candidates"])
    assert all(candidate["complete_quality_bundle"] is False for candidate in decision["candidates"])


if __name__ == "__main__":
    test_rejected_components_cannot_be_ensembled_into_runtime_authority()
    test_decision_preserves_component_scope()
    print("general provider candidate decision v2: ok")
