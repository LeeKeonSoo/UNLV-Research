from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence.livecodebench_pilot import (
    BenchmarkTask,
    select_stratified_pilot,
    screen_lexical_overlap,
)


def _task(platform: str, difficulty: str, index: int) -> BenchmarkTask:
    return BenchmarkTask(
        question_id=f"{platform}-{difficulty}-{index}",
        contest_date=date(2025, 1, (index % 28) + 1),
        platform=platform,
        difficulty=difficulty,
        prompt_text=f"Solve unique problem {platform} {difficulty} {index}",
    )


def test_select_stratified_pilot_balances_every_cell() -> None:
    # Given: more than eight tasks in every platform/difficulty cell.
    tasks = tuple(
        _task(platform, difficulty, index)
        for platform in ("atcoder", "leetcode")
        for difficulty in ("easy", "medium", "hard")
        for index in range(12)
    )

    # When: the frozen pilot selector is applied twice.
    first = select_stratified_pilot(tasks, per_cell=8, selection_seed="pilot-v1")
    second = select_stratified_pilot(tuple(reversed(tasks)), per_cell=8, selection_seed="pilot-v1")

    # Then: selection is order-independent and contains eight tasks per cell.
    assert first == second
    assert len(first) == 48
    assert {
        (platform, difficulty): sum(
            task.platform == platform and task.difficulty == difficulty for task in first
        )
        for platform in ("atcoder", "leetcode")
        for difficulty in ("easy", "medium", "hard")
    } == {
        (platform, difficulty): 8
        for platform in ("atcoder", "leetcode")
        for difficulty in ("easy", "medium", "hard")
    }


def test_screen_lexical_overlap_reports_matching_training_record() -> None:
    # Given: one training row copies several benchmark-specific eight-token sequences.
    benchmark = (
        BenchmarkTask(
            question_id="42",
            contest_date=date(2025, 2, 1),
            platform="atcoder",
            difficulty="hard",
            prompt_text=(
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
            ),
        ),
    )
    training = (
        ("clean", "unrelated implementation text"),
        (
            "copied",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda helper",
        ),
    )

    # When: lexical overlap is screened at eight tokens.
    result = screen_lexical_overlap(training, benchmark, ngram_size=8)

    # Then: the copied row and benchmark question are identified.
    assert result.candidate_count == 1
    assert result.candidates[0].training_id == "copied"
    assert result.candidates[0].question_id == "42"


def test_screen_lexical_overlap_ignores_one_generic_ngram() -> None:
    # Given: code and a benchmark share only one generic eight-token sequence.
    benchmark = (
        BenchmarkTask(
            question_id="generic",
            contest_date=date(2025, 2, 1),
            platform="leetcode",
            difficulty="easy",
            prompt_text="for index in range left right return value unique tail",
        ),
    )
    training = (("ordinary", "for index in range left right return value other code"),)

    # When: lexical overlap is screened.
    result = screen_lexical_overlap(training, benchmark, ngram_size=8)

    # Then: the isolated generic match is not treated as a contamination candidate.
    assert result.candidate_count == 0


if __name__ == "__main__":
    test_select_stratified_pilot_balances_every_cell()
    test_screen_lexical_overlap_reports_matching_training_record()
    test_screen_lexical_overlap_ignores_one_generic_ngram()
    print("[code-livecodebench-pilot] selection and overlap contracts: pass")
