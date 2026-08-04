#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_source_pool_contract import ContrastiveSourcePoolProtocol
from contrastive_source_pool_materialization import build_source_pools


def _protocol() -> ContrastiveSourcePoolProtocol:
    protocol = ContrastiveSourcePoolProtocol.model_validate_json(
        (ROOT / "protocols" / "contrastive_operating_point_source_pool_v1.json").read_text()
    )
    return protocol.model_copy(
        update={"sampling": protocol.sampling.model_copy(update={"records_per_source_after_stage_a": 100})}
    )


def _rows(source_id: str) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"{source_id}-{index}",
            "text": f"Substantive development document {source_id} number {index} with stable payload.",
        }
        for index in range(120)
    ]


def test_materialization_emits_one_common_baseline_and_one_shared_eligible_pool() -> None:
    protocol = _protocol()
    rows = {source.source_id: _rows(source.source_id) for source in protocol.sources}

    baseline, eligible, manifest = build_source_pools(protocol, rows, lambda text: len(text.split()))

    assert len(baseline) == 300
    assert len(eligible) == 600
    assert manifest["baseline_record_overlap_count"] == 0
    assert manifest["baseline_normalized_text_overlap_count"] == 0
    assert manifest["normal_and_hard_share_eligible_pool"] is True
    assert {row["pool_role"] for row in baseline} == {"common_baseline"}
    assert {row["pool_role"] for row in eligible} == {"eligible_arm"}
    assert all(row["stage_a_reason_codes"] == [] for row in baseline + eligible)


if __name__ == "__main__":
    test_materialization_emits_one_common_baseline_and_one_shared_eligible_pool()
    print("[contrastive-source-pool-materialization-v1] common baseline and shared pool: pass")
