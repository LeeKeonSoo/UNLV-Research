#!/usr/bin/env python3
"""Regression tests for operational curation-decision contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _load_decision_module() -> ModuleType:
    path = PROJECT_DIR / "27_build_curation_decision_report.py"
    spec = importlib.util.spec_from_file_location("curation_decision_report_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_a_evidence(
    *,
    stage_a_records: int,
    selected_records: int,
    baseline_candidates: int,
) -> Dict[str, Any]:
    return {
        "stage": "A",
        "status": "pass",
        "evidence": {
            "stage_a_records": stage_a_records,
            "selected_records": selected_records,
            "stage_a_candidate_records_excluding_selected": baseline_candidates,
        },
    }


def main() -> int:
    module = _load_decision_module()

    stale_positive = module._decision_from_evidence(
        {"stage_c": {"passed": True}},
        {"protocol_status": "certified_ready"},
        {"certification_claim_allowed": True},
        _stage_a_evidence(stage_a_records=1, selected_records=1, baseline_candidates=0),
    )
    assert stale_positive["decision"] == "insufficient_usable_data", stale_positive
    assert stale_positive["operational_action"] == "insufficient_usable_data", stale_positive
    assert stale_positive["certification_claim_allowed"] is False, stale_positive

    supported_positive = module._decision_from_evidence(
        {"stage_c": {"passed": True}},
        {"protocol_status": "certified_ready"},
        {"certification_claim_allowed": True},
        _stage_a_evidence(stage_a_records=100, selected_records=20, baseline_candidates=80),
    )
    assert supported_positive["decision"] == "accepted_for_training", supported_positive
    assert supported_positive["operational_action"] == "accept", supported_positive
    assert supported_positive["certification_claim_allowed"] is True, supported_positive

    print("[decision-contract] UC-10 abstention overrides stale positive Stage-C evidence: pass")
    print("[decision-contract] sufficient usable data preserves supported acceptance: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
