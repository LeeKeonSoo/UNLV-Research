from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence.livecodebench_evaluate import summarize_strata


def test_summarize_strata_preserves_platform_and_difficulty() -> None:
    # Given: mixed pass outcomes across two benchmark strata.
    rows = (
        {"platform": "atcoder", "difficulty": "hard", "passed": True},
        {"platform": "atcoder", "difficulty": "hard", "passed": False},
        {"platform": "leetcode", "difficulty": "easy", "passed": True},
    )

    # When: stratum metrics are summarized.
    summary = summarize_strata(rows)

    # Then: each cell reports its own denominator and pass rate.
    assert summary == (
        {
            "platform": "atcoder",
            "difficulty": "hard",
            "task_count": 2,
            "pass_count": 1,
            "pass_rate": 0.5,
        },
        {
            "platform": "leetcode",
            "difficulty": "easy",
            "task_count": 1,
            "pass_count": 1,
            "pass_rate": 1.0,
        },
    )


if __name__ == "__main__":
    test_summarize_strata_preserves_platform_and_difficulty()
    print("[code-livecodebench-evaluate] stratum summary: pass")
