from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence import livecodebench_summary


def test_build_summary_reports_neutral_independent_transfer() -> None:
    # Given: three arms with identical outcomes but different generated programs.
    base = (False, True, False, True)
    raw = (False, True, False, True)
    curated = (False, True, False, True)
    code_differences = {
        "base_vs_raw": 3,
        "base_vs_curated": 2,
        "raw_vs_curated": 1,
    }

    # When: the paired pilot evidence is summarized.
    summary = livecodebench_summary.build_summary(
        base=base,
        raw=raw,
        curated=curated,
        code_differences=code_differences,
    )

    # Then: no transfer gain is claimed despite changed generated code.
    assert summary["status"] == "completed_no_independent_transfer_gain"
    assert summary["pass_at_1"] == {
        "base_no_update": 0.5,
        "raw_full_natural": 0.5,
        "curated_v2_natural": 0.5,
    }
    assert summary["paired_curated_vs_raw"] == {
        "curated_wins": 0,
        "curated_losses": 0,
        "ties": 4,
        "exact_two_sided_p": 1.0,
    }
    assert summary["generation_code_differences"] == code_differences
    assert summary["claim"] == "independent_transfer_not_demonstrated"


if __name__ == "__main__":
    test_build_summary_reports_neutral_independent_transfer()
    print("[code-livecodebench-summary] paired evidence: pass")
