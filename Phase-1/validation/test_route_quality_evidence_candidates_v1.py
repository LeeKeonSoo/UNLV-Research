#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_quality_evidence_candidates import (
    CandidateEvidenceContractError,
    audit_candidate_registry,
    evaluate_registered_candidate,
)


MANIFEST = ROOT / "configs" / "route_quality_evidence_candidates_v1.json"
ROUTE_TEXTS = {
    "general_prose": "The experiment measures glacier density because gravity deforms the ice over time. Scientists compare field instruments with satellite images.",
    "code_artifact": "#!/usr/bin/env python3\nimport math\ndef area(radius):\n    return math.pi * radius * radius",
    "mathematical_content": "The theorem follows from the matrix equation. Proof: use \\frac{1}{2} and \\sum_i x_i.",
    "technical_documentation": "API Reference\nParameters\nvalue: the input integer.\nReturns\nThe transformed integer result.",
    "conversation": "Ava: Are you coming after class?\nBen: Yes, I will meet you outside.\nAva: Great, I will wait near the library.",
    "instruction": "Step 1: install the package.\nStep 2: open the configuration file.\nStep 3: run the validation command.",
    "table_structured_data": "| Model | Score |\n| --- | --- |\n| Base | 42.1 |",
}
EXPECTED_HEADS = {
    "general_prose": ("missing", "indeterminate"),
    "code_artifact": ("missing", "indeterminate"),
    "mathematical_content": ("indeterminate", "indeterminate"),
    "technical_documentation": ("missing", "missing"),
    "conversation": ("missing", "missing"),
    "instruction": ("missing", "missing"),
    "table_structured_data": ("missing", "missing"),
}


def main() -> int:
    audit = audit_candidate_registry(MANIFEST)
    assert audit["ready_routes"] == []
    assert audit["summary"] == {
        "routes": 7,
        "evidence_ready_heads": 0,
        "blocked_heads": 4,
        "unsupported_heads": 10,
    }
    assert audit["runtime_authorized"] is False
    for route, text in ROUTE_TEXTS.items():
        result = evaluate_registered_candidate(text, MANIFEST)
        assert result["decision"]["decision"] == "abstain_retain"
        assert result["decision"]["reason_code"] == "quality_evidence_incomplete"
        assert result["routed_routes"] == [route]
        assert result["head_outcomes"][route] == {
            "substantive_payload": EXPECTED_HEADS[route][0],
            "route_specific_evidence": EXPECTED_HEADS[route][1],
        }
        assert result["runtime_authorized"] is False
        assert result["selector_inputs"] == ["chunk_text", "frozen_candidate_registry"]
        assert result["benchmark_outcomes_read"] is False
        assert result["utility_read"] is False

    unknown = evaluate_registered_candidate("@@@ ### 12345 === ???", MANIFEST)
    assert unknown["decision"]["reason_code"] == "quality_routing_unknown"
    assert unknown["head_outcomes"] == {}

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["routes"]["code_artifact"]["route_specific_evidence"]["artifact_sha256"] = "0" * 64
    with tempfile.TemporaryDirectory() as directory:
        tampered = Path(directory) / "manifest.json"
        tampered.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            evaluate_registered_candidate(ROUTE_TEXTS["code_artifact"], tampered)
        except CandidateEvidenceContractError as error:
            assert "artifact SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("Tampered evidence artifact must fail closed")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    head = manifest["routes"]["general_prose"]["route_specific_evidence"]
    head["outcome"] = "negative"
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "manifest.json"
        invalid.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            evaluate_registered_candidate(ROUTE_TEXTS["general_prose"], invalid)
        except CandidateEvidenceContractError as error:
            assert "blocked or unsupported evidence cannot declare pass or negative" in str(error)
        else:
            raise AssertionError("Blocked evidence must not manufacture deletion authority")
    print("[route-quality-candidates-v1] frozen route evidence fails closed: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
