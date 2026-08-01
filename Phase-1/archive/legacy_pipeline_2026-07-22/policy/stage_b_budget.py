from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Literal, TypeVar, TypedDict


RecordT = TypeVar("RecordT")


class StageBBudgetConfig(TypedDict, total=False):
    max_word_count: int


class StageBProfile(TypedDict, total=False):
    stage_b_budget: StageBBudgetConfig


@dataclass(frozen=True, slots=True)
class StageBBudget:
    mode: Literal["retain_all", "word_budget"]
    binding: bool
    word_limit: int | None


def resolve_stage_b_budget(profile: StageBProfile, *, total_word_count: int) -> StageBBudget:
    raw = profile.get("stage_b_budget") or {}
    declared_limit = int(raw.get("max_word_count") or 0)
    if declared_limit <= 0 or declared_limit >= max(0, total_word_count):
        return StageBBudget(mode="retain_all", binding=False, word_limit=None)
    return StageBBudget(mode="word_budget", binding=True, word_limit=declared_limit)


def fit_word_budget(
    records: Sequence[RecordT],
    *,
    word_count: Callable[[RecordT], int],
    word_limit: int,
) -> list[RecordT]:
    selected: list[RecordT] = []
    used_words = 0
    for record in records:
        record_words = max(0, word_count(record))
        if used_words + record_words > word_limit:
            continue
        selected.append(record)
        used_words += record_words
    return selected
