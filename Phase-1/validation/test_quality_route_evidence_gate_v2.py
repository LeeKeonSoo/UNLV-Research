#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_evidence_gate import RouteEvidenceGate, evaluate_route_evidence_gate


MANIFEST = ROOT / "configs" / "quality_route_evidence_gate_v2.json"
EXPECTED_STATUS = {
    "general_prose": "blocked_source_transfer",
    "code_artifact": "blocked_source_transfer",
    "mathematical_content": "blocked_source_transfer",
}


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_route_gates_match_observed_evidence() -> None:
    payload = load_manifest()
    assert payload["schema_version"] == "quality-route-evidence-gate-v2"
    assert payload["quality_heads"] == [
        "substantive_payload",
        "route_specific_evidence",
    ]
    assert payload["routing_precondition"] == {
        "name": "route_confidence",
        "owner": "content_router_v2",
        "quality_evidence": False,
        "may_authorize_removal": False,
    }
    assert payload["runtime_activation"] is False

    routes = payload["routes"]
    assert isinstance(routes, list)
    assert {route["route"] for route in routes} == set(EXPECTED_STATUS)
    for route in routes:
        result = evaluate_route_evidence_gate(RouteEvidenceGate.from_mapping(route))
        assert result.status == EXPECTED_STATUS[result.route]
        assert result.runtime_authorized is False
        assert result.failed_gates
        assert route["decision"] == result.status


def test_gate_forbids_quality_v1_coherence_and_runtime_feedback() -> None:
    payload = load_manifest()
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "coherence_completeness" not in serialized
    for forbidden in payload["forbidden_runtime_inputs"]:
        assert forbidden in {
            "benchmark",
            "nll",
            "utility",
            "source_reputation",
            "domain_quota",
            "target_retention_fraction",
        }


def test_local_frozen_artifact_hashes_are_exact() -> None:
    routes = load_manifest()["routes"]
    assert isinstance(routes, list)
    path_keys = (
        "frozen_bundle",
        "calibration_report",
        "frozen_decision",
        "calibration_contract",
        "provider_candidate_decision",
    )
    for route in routes:
        observed = route["observed_evidence"]
        for path_key in path_keys:
            relative = observed.get(path_key)
            if relative is None:
                continue
            expected = observed[f"{path_key}_sha256"]
            artifact = ROOT / relative
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            assert actual == expected, f"{route['route']}:{path_key}"


def test_general_route_records_independent_provider_rejections() -> None:
    general = next(route for route in load_manifest()["routes"] if route["route"] == "general_prose")
    observed = general["observed_evidence"]

    assert observed["provider_candidate_decision"] == "configs/general_provider_candidate_decision_v2.json"
    assert observed["additional_rejected_providers"] == [
        "mlfoundations/fasttext-oh-eli5",
        "HuggingFaceFW/fineweb-edu-classifier",
    ]


def test_only_all_required_gates_can_become_evidence_ready() -> None:
    ready = RouteEvidenceGate(
        route="general_prose",
        artifacts_frozen=True,
        source_and_hash_disjoint=True,
        strict_source_transfer=True,
        adversarial_and_format_fixtures=True,
        provider_bias_stress=True,
        route_holdout_stress=True,
        external_results_hidden=True,
    )
    result = evaluate_route_evidence_gate(ready)
    assert result.status == "evidence_ready_candidate"
    assert result.runtime_authorized is False
    assert result.failed_gates == ()


if __name__ == "__main__":
    test_frozen_route_gates_match_observed_evidence()
    test_gate_forbids_quality_v1_coherence_and_runtime_feedback()
    test_local_frozen_artifact_hashes_are_exact()
    test_general_route_records_independent_provider_rejections()
    test_only_all_required_gates_can_become_evidence_ready()
    print("quality route evidence gate v2: ok")
