#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_historical_proxy_forensics import build_report


def row(chunk_uid: str, accepted: bool) -> dict[str, object]:
    selection: dict[str, object] = {
        "operational_priority": {
            "length_support": 0.8,
            "structural_richness": 0.6,
            "lexical_or_identifier_diversity": 0.4,
            "pass_through_assignment_ratio": 0.0,
            "score": 0.81 if accepted else 0.79,
        }
    }
    if not accepted:
        selection["removed_reason"] = "operational_priority_below_frozen_threshold"
    return {
        "chunk_uid": chunk_uid,
        "text": "def useful_function():\n    return 1\n",
        "stage_b_selection": selection,
        "stage_c_policy_metadata": {"path": "src/useful.py"},
    }


def main() -> int:
    report = build_report([row("kept", True)], [row("removed", False)], [row("removed", True)])
    assert report["historical_groups"]["priority_rejected"]["chunks"] == 1
    assert report["historical_groups"]["hard_gate_rejected"] == 0
    assert report["cross_version_overlap"] == {
        "historical_priority_rejected_chunks": 1,
        "also_retained_by_current_v3": 1,
        "retained_share": 1.0,
    }
    print("[historical-proxy-forensics] proxy boundary fixture: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
