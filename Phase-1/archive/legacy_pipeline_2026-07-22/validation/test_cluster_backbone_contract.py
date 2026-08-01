#!/usr/bin/env python3
"""Regression tests for the Stage-C semantic cluster backbone audit."""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from policy.subsets import _cluster_backbone_audit


CLUSTER_TERMS = (
    "astronomy",
    "biology",
    "chemistry",
    "economics",
    "geography",
    "history",
    "literature",
    "mathematics",
)


def _connection(*, coherent: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_uid TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            cluster_id INTEGER,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            source TEXT NOT NULL,
            input_source TEXT NOT NULL
        )
        """
    )
    for cluster_id, term in enumerate(CLUSTER_TERMS):
        for row_id in range(8):
            topic = term if coherent else "shared"
            text = f"{topic} focused lesson explains concepts examples practice details number {row_id}"
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{cluster_id:02d}-{row_id:02d}",
                    "fixture",
                    cluster_id,
                    text,
                    '{"domain": "single-domain"}',
                    "single-source",
                    "single-input",
                ),
            )
    conn.commit()
    return conn


def main() -> int:
    cluster_counts = Counter({cluster_id: 8 for cluster_id in range(len(CLUSTER_TERMS))})
    coherent = _cluster_backbone_audit(
        _connection(coherent=True),
        dataset="fixture",
        original_clusters=cluster_counts,
        seed=1729,
    )
    assert coherent["passed"] is True, coherent
    assert coherent["lexical_separation_pass"] is True, coherent
    assert coherent["within_gt_between_fraction"] >= 0.55, coherent
    assert coherent["anchor_purity_role"] == "diagnostic_only", coherent

    collapsed = _cluster_backbone_audit(
        _connection(coherent=False),
        dataset="fixture",
        original_clusters=cluster_counts,
        seed=1729,
    )
    assert collapsed["anchor_purity_pass"] is True, collapsed
    assert collapsed["lexical_separation_pass"] is False, collapsed
    assert collapsed["passed"] is False, collapsed
    assert "pairwise_lexical_separation_failed" in collapsed["failure_reasons"], collapsed
    print("[cluster-backbone-contract] pairwise separation and diagnostic-only anchors: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
