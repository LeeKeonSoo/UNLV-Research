#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_reference_distribution_calibration import build_calibration


def _row(record_id: str, tier: str, repository: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "text": f"def {record_id.replace('-', '_')}():\n    return 1\n",
        "partition": {"source_tier": tier, "repository_identity": repository},
    }


def test_reference_distribution_calibration_uses_repository_disjoint_holdout() -> None:
    rows = [
        _row("reference-a", "known_high_quality_reference", "repo-a"),
        _row("reference-b", "known_high_quality_reference", "repo-b"),
        _row("reference-c", "known_high_quality_reference", "repo-c"),
        _row("raw-a", "raw_like", "raw-repo-a"),
        _row("raw-b", "raw_like", "raw-repo-b"),
    ]

    train, calibration, report = build_calibration(rows, held_out_repository_count=1, split_salt="fixture")

    assert len(train) == 2
    assert len(calibration) == 2
    assert {row["source_role_label"] for row in calibration} == {
        "reference_distribution_member",
        "raw_like_nonmember",
    }
    assert report["repository_overlap"] == []
    assert report["summary"]["calibration_positive_records"] == 1
    assert report["summary"]["calibration_negative_records"] == 1


if __name__ == "__main__":
    test_reference_distribution_calibration_uses_repository_disjoint_holdout()
    print("[reference-distribution-calibration] repository-disjoint source-role labels: pass")
