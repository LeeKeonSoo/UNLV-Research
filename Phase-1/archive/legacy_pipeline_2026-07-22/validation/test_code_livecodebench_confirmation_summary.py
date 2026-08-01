from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence import livecodebench_confirmation_summary


def test_build_summary_marks_small_multiseed_external_difference_inconclusive() -> None:
    # Given: two aligned seeds with one curated-only correctness transition.
    raw_by_seed = {
        101: (False, True),
        131: (False, False),
    }
    curated_by_seed = {
        101: (True, True),
        131: (False, False),
    }

    # When: frozen external evaluations are aggregated before any policy change.
    summary = livecodebench_confirmation_summary.build_summary(
        raw_by_seed=raw_by_seed,
        curated_by_seed=curated_by_seed,
    )

    # Then: a small directional change is reported without a transfer-gain claim.
    assert summary["status"] == "completed_multiseed_external_transfer_inconclusive"
    assert summary["raw_mean_pass_rate"] == 0.25
    assert summary["curated_mean_pass_rate"] == 0.5
    assert summary["mean_pass_rate_delta"] == 0.25
    assert summary["pooled_paired"] == {
        "curated_wins": 1,
        "curated_losses": 0,
        "ties": 3,
        "exact_two_sided_p": 1.0,
    }


if __name__ == "__main__":
    test_build_summary_marks_small_multiseed_external_difference_inconclusive()
    print("[code-livecodebench-confirmation-summary] aggregate contract: pass")
