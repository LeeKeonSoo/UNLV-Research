from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from paper_evidence.livecodebench_runner import (
    DEFAULT_TRAINING_OUTPUT,
    StopOnTokenSequences,
    trim_generic_base_completion,
)


def test_trim_generic_base_completion_stops_before_next_question() -> None:
    # Given: a base-model completion continues into the next few-shot delimiter.
    completion = "import sys\nprint(sys.stdin.read())\n\n### Question\nAnother task"

    # When: the official GenericBase completion boundary is applied.
    trimmed = trim_generic_base_completion(completion)

    # Then: executable code remains and the next question is removed.
    assert trimmed == "import sys\nprint(sys.stdin.read())\n"


def test_stop_on_token_sequences_ends_at_official_delimiter() -> None:
    # Given: generated token IDs end with one frozen stop sequence.
    criterion = StopOnTokenSequences(((30, 40), (99,)))
    matching = torch.tensor([[10, 20, 30, 40]])
    nonmatching = torch.tensor([[10, 20, 30, 41]])

    # When: the stopping criterion observes both sequences.
    stopped = criterion(matching, torch.empty(0))
    continued = criterion(nonmatching, torch.empty(0))

    # Then: only the official delimiter terminates decoding.
    assert stopped is True
    assert continued is False


def test_default_training_output_uses_current_framework_rerun() -> None:
    # Given: the LiveCodeBench runner's default training artifact root.
    training_output = DEFAULT_TRAINING_OUTPUT

    # When: a caller does not explicitly override that root.
    selected_root = training_output.name

    # Then: generation loads current-framework rather than legacy adapters.
    assert selected_root == "current_framework_rerun"


if __name__ == "__main__":
    test_trim_generic_base_completion_stops_before_next_question()
    test_stop_on_token_sequences_ends_at_official_delimiter()
    test_default_training_output_uses_current_framework_rerun()
    print("[code-livecodebench-runner] completion boundary: pass")
