#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from preflight_calibrated_selector import preflight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibrated_selector_preflight() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference.jsonl"
        calibration = root / "calibration.jsonl"
        scope_audit = root / "scope_audit.json"
        external_plan = root / "external_plan.json"
        for path in (reference, calibration, scope_audit, external_plan):
            path.write_text('{"fixture": true}\n', encoding="utf-8")
        contract = root / "selector.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": "calibrated-selector-contract-v1",
                    "status": "frozen_candidate",
                    "profile_id": "calibrated_selector_template_v1",
                    "selection_hypothesis": "reference_distribution_membership_for_declared_code_scope",
                    "score_direction": "higher_means_more_similar_to_frozen_reference_distribution",
                    "selector_boundary": {"utility_read": False, "benchmark_outcomes_read": False, "target_token_fraction_read": False},
                    "artifacts": {
                        "reference_data": {"path": str(reference), "sha256": _sha256(reference)},
                        "held_out_calibration": {"path": str(calibration), "sha256": _sha256(calibration)},
                        "scope_audit": {"path": str(scope_audit), "sha256": _sha256(scope_audit)},
                        "external_validation_plan": {"path": str(external_plan), "sha256": _sha256(external_plan)},
                    },
                    "false_positive_audit": {"status": "passed"},
                    "activation": {"status": "not_implemented"},
                }
            ),
            encoding="utf-8",
        )

        report = preflight(contract)

        assert report["status"] == "ready_for_selector_implementation"
        assert report["pending_gates"] == ["selector implementation is not present"]


if __name__ == "__main__":
    test_calibrated_selector_preflight()
    print("[calibrated-selector-preflight] frozen-evidence boundary: pass")
