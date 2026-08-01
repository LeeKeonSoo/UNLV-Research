from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Final, Sequence

TOKEN_PATTERN: Final = re.compile(r"[a-z_][a-z0-9_]*|\d+|[^\s]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    question_id: str
    contest_date: date
    platform: str
    difficulty: str
    prompt_text: str


@dataclass(frozen=True, slots=True)
class OverlapCandidate:
    training_id: str
    question_id: str
    shared_ngram_count: int
    containment: float


@dataclass(frozen=True, slots=True)
class OverlapResult:
    candidates: tuple[OverlapCandidate, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class InsufficientCellError(ValueError):
    platform: str
    difficulty: str
    available: int
    required: int

    def __str__(self) -> str:
        return (
            f"{self.platform}/{self.difficulty} has {self.available} tasks; "
            f"{self.required} required"
        )


def _selection_key(task: BenchmarkTask, selection_seed: str) -> str:
    payload = f"{selection_seed}\0{task.platform}\0{task.difficulty}\0{task.question_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_stratified_pilot(
    tasks: Sequence[BenchmarkTask],
    *,
    per_cell: int,
    selection_seed: str,
) -> tuple[BenchmarkTask, ...]:
    cells = sorted({(task.platform, task.difficulty) for task in tasks})
    selected: list[BenchmarkTask] = []
    for platform, difficulty in cells:
        candidates = tuple(
            task
            for task in tasks
            if task.platform == platform and task.difficulty == difficulty
        )
        if len(candidates) < per_cell:
            raise InsufficientCellError(platform, difficulty, len(candidates), per_cell)
        ranked = sorted(candidates, key=lambda task: _selection_key(task, selection_seed))
        selected.extend(ranked[:per_cell])
    return tuple(sorted(selected, key=lambda task: task.question_id))


def _ngrams(text: str, ngram_size: int) -> set[tuple[str, ...]]:
    tokens = tuple(token.lower() for token in TOKEN_PATTERN.findall(text))
    return {
        tokens[index : index + ngram_size]
        for index in range(max(0, len(tokens) - ngram_size + 1))
    }


def screen_lexical_overlap(
    training_records: Sequence[tuple[str, str]],
    benchmark_tasks: Sequence[BenchmarkTask],
    *,
    ngram_size: int,
) -> OverlapResult:
    benchmark_ngrams = {
        task.question_id: _ngrams(task.prompt_text, ngram_size) for task in benchmark_tasks
    }
    candidates: set[OverlapCandidate] = set()
    for training_id, text in training_records:
        training_ngrams = _ngrams(text, ngram_size)
        for question_id, task_ngrams in benchmark_ngrams.items():
            shared = len(training_ngrams & task_ngrams)
            denominator = min(len(training_ngrams), len(task_ngrams))
            containment = shared / denominator if denominator else 0.0
            if shared >= 4 and containment >= 0.5:
                candidates.add(
                    OverlapCandidate(
                        training_id=training_id,
                        question_id=question_id,
                        shared_ngram_count=shared,
                        containment=round(containment, 6),
                    )
                )
    return OverlapResult(
        candidates=tuple(
            sorted(candidates, key=lambda item: (item.training_id, item.question_id))
        )
    )
