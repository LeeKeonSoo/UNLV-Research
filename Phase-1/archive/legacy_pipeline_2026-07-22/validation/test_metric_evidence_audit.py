#!/usr/bin/env python3
"""Regression checks for honest metric-evidence classification."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    audit = load_json(PROJECT_DIR / "configs" / "metric_evidence_audit.json")
    assert audit["schema_version"] == "metric-evidence-audit-v1"
    assert audit["known_citation_gap"]["status"] == "incomplete"
    review = audit["human_or_llm_review"]
    assert review["required_for_stage_b_approval"] is False
    assert review["required_for_stage_c_entry"] is False
    assert review["may_tune_or_promote_selector"] is False
    components = audit["components"]
    project_parameters = [
        name
        for name, row in components.items()
        if "project-specific" in str(row.get("parameter_origin") or "")
    ]
    assert project_parameters, components
    assert all(components[name]["evidence_class"] == "project_hypothesis_frozen" for name in project_parameters)
    assert all(row.get("required_next_evidence") for row in components.values())
    print("[metric-evidence-audit] citations, project hypotheses, and optional review are separated: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
