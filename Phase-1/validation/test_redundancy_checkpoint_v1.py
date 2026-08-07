from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redundancy_checkpoint import load_or_build_redundancy
from redundancy_equivalence import RedundancyMode
from redundancy_v2 import RedundancySettings


def test_redundancy_checkpoint_reuses_only_an_identical_input() -> None:
    rows = [
        {"chunk_uid": "a", "text": "alpha beta gamma", "token_proxy": 3},
        {"chunk_uid": "b", "text": "alpha beta gamma", "token_proxy": 3},
        {"chunk_uid": "c", "text": "distinct payload", "token_proxy": 2},
    ]
    settings = RedundancySettings()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "redundancy.json"

        first = load_or_build_redundancy(
            rows,
            mode=RedundancyMode.NORMAL,
            settings=settings,
            checkpoint_path=path,
        )
        second = load_or_build_redundancy(
            rows,
            mode=RedundancyMode.NORMAL,
            settings=settings,
            checkpoint_path=path,
        )
        changed = load_or_build_redundancy(
            [*rows[:-1], {**rows[-1], "text": "changed payload"}],
            mode=RedundancyMode.NORMAL,
            settings=settings,
            checkpoint_path=path,
        )

        assert first.checkpoint_hit is False
        assert second.checkpoint_hit is True
        assert changed.checkpoint_hit is False
        assert second.identity_sha256 == first.identity_sha256
        assert changed.identity_sha256 != first.identity_sha256
        assert second.result.audit == first.result.audit
        assert second.result.plan.families == first.result.plan.families


if __name__ == "__main__":
    test_redundancy_checkpoint_reuses_only_an_identical_input()
    print("[redundancy-checkpoint-v1] identity and reuse: pass")
