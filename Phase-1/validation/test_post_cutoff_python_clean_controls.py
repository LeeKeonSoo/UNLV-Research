#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_post_cutoff_python_clean_controls import RepoSpec, build_control_records


def test_builder_keeps_only_disjoint_substantive_complete_python_sources() -> None:
    # Given: two post-cutoff repositories with valid, generated, empty, and duplicate files.
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first.mkdir()
        second.mkdir()
        good = "def add(left, right):\n    return left + right\n"
        overlap = "def overlap():\n    return 1\n"
        (first / "good.py").write_text(good, encoding="utf-8")
        (first / "generated.py").write_text("# generated file - do not edit\nvalue = 1\n", encoding="utf-8")
        (first / "empty.py").write_text('"""Only a notice."""\npass\n', encoding="utf-8")
        (first / "overlap.py").write_text(overlap, encoding="utf-8")
        (second / "duplicate.py").write_text(good, encoding="utf-8")
        (second / "unique.py").write_text("class Adapter:\n    pass\n", encoding="utf-8")

        # When: the clean-control builder applies frozen structural and disjointness gates.
        rows, counts = build_control_records(
            (
                RepoSpec("org/first", first, "a" * 40),
                RepoSpec("org/second", second, "b" * 40),
            ),
            frozenset({" ".join(overlap.split())}),
            lambda text: len(text.split()),
        )

    # Then: only one representative per new substantive source remains.
    assert [row["record_id"] for row in rows] == [
        "org/first::good.py",
        "org/second::unique.py",
    ]
    assert all(row["language"] == {"code": "python", "confidence": 1.0, "declaration": "source_row"} for row in rows)
    assert counts == {
        "python_files_seen": 6,
        "generated_marker_excluded": 1,
        "non_substantive_excluded": 1,
        "candidate_hash_overlap_excluded": 1,
        "cross_repository_duplicate_excluded": 1,
        "retained": 2,
    }


if __name__ == "__main__":
    test_builder_keeps_only_disjoint_substantive_complete_python_sources()
    print("[post-cutoff-python-controls] disjoint substantive source gate: pass")
