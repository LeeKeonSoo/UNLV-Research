#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inline_license_header_compaction import build_plan, materialize_candidate_plan


def test_prefix_license_header_candidate_removes_only_header_and_preserves_payload() -> None:
    header = "# Copyright 2025 Example Foundation\n# SPDX-License-Identifier: Apache-2.0\n# Licensed under the Apache License, Version 2.0"
    payload = "def add(left, right):\n    return left + right\n\nprint(add(2, 3))"
    rows = [{"chunk_uid": "code::0000", "text": f"{header}\n\n{payload}", "token_proxy": 30}]

    plan = build_plan(rows, minimum_residual_tokens=8)
    result = materialize_candidate_plan(rows, plan)

    assert plan["candidate_header_removals"] == 1
    assert result["records"][0]["text"] == payload
    assert result["transformations"][0]["reason_code"] == "inline_license_header_removed"
    assert result["transformations"][0]["header_token_proxy"] > 0


def test_license_reference_after_payload_is_not_a_prefix_header_candidate() -> None:
    text = "def explain_license():\n    return 'Copyright holders should read the license text.'\n\n# Copyright 2025 Example Foundation\n# SPDX-License-Identifier: Apache-2.0"
    rows = [{"chunk_uid": "code::0000", "text": text}]

    plan = build_plan(rows, minimum_residual_tokens=8)
    result = materialize_candidate_plan(rows, plan)

    assert plan["candidate_header_removals"] == 0
    assert result["records"][0]["text"] == text


def test_license_only_chunk_is_not_rewritten_without_payload() -> None:
    text = "# Copyright 2025 Example Foundation\n# SPDX-License-Identifier: Apache-2.0"
    rows = [{"chunk_uid": "code::0000", "text": text}]

    plan = build_plan(rows, minimum_residual_tokens=8)

    assert plan["candidate_header_removals"] == 0
    assert plan["blocked_no_payload_chunks"] == ["code::0000"]


def test_copyright_attribution_without_explicit_license_evidence_is_retained() -> None:
    text = "# Copyright 2025 Example Foundation\n\ndef add(left, right):\n    return left + right\n\nprint(add(2, 3))"
    rows = [{"chunk_uid": "code::0000", "text": text}]

    plan = build_plan(rows, minimum_residual_tokens=8)

    assert plan["candidate_header_removals"] == 0


if __name__ == "__main__":
    test_prefix_license_header_candidate_removes_only_header_and_preserves_payload()
    test_license_reference_after_payload_is_not_a_prefix_header_candidate()
    test_license_only_chunk_is_not_rewritten_without_payload()
    test_copyright_attribution_without_explicit_license_evidence_is_retained()
    print("[inline-license-header-compaction] text-only prefix-header boundary: pass")
