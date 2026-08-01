#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inline_license_comment_block_compaction import build_plan, materialize_candidate_plan


def test_inline_explicit_license_comment_block_is_removed_without_removing_payload() -> None:
    text = "def add(left, right):\n    return left + right\n\n# SPDX-License-Identifier: Apache-2.0\n# Licensed under the Apache License, Version 2.0\n\nprint(add(2, 3))"
    plan = build_plan([{"chunk_uid": "code::0000", "text": text}], minimum_residual_tokens=8)
    result = materialize_candidate_plan([{"chunk_uid": "code::0000", "text": text}], plan)

    assert plan["candidate_block_removals"] == 1
    assert "SPDX-License-Identifier" not in result["records"][0]["text"]
    assert "def add" in result["records"][0]["text"]
    assert result["transformations"][0]["reason_code"] == "inline_license_comment_block_removed"
    assert result["transformations"][0]["block_token_proxy"] > 0


def test_copyright_only_comment_block_is_retained() -> None:
    text = "def add(left, right):\n    return left + right\n\n# Copyright 2025 Example Foundation\n\nprint(add(2, 3))"
    plan = build_plan([{"chunk_uid": "code::0000", "text": text}], minimum_residual_tokens=8)

    assert plan["candidate_block_removals"] == 0


if __name__ == "__main__":
    test_inline_explicit_license_comment_block_is_removed_without_removing_payload()
    test_copyright_only_comment_block_is_retained()
    print("[inline-license-comment-block-compaction] text-only inline block boundary: pass")
