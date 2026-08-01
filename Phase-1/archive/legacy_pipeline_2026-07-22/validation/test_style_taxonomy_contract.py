#!/usr/bin/env python3
"""Regression test for aligned Stage-B and Stage-C style taxonomy."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from collections import Counter

from policy.subsets import (
    _structured_text_relief,
    _style_bucket_from_scored_record,
    _style_bucket_from_text,
    _style_taxonomy_alignment_diagnostic,
)
from signals.core import style_bucket_from_text


def main() -> int:
    record = {
        "core_metrics": {
            "structural_validity_gate": {
                "details": {"style_bucket": "technical_reference"}
            }
        },
        "provenance": {
            "text_preview": "This truncated preview looks like ordinary general prose."
        },
    }
    scored_style = _style_bucket_from_scored_record(record)
    selector_style = _structured_text_relief(record)["style_bucket"]
    assert scored_style == "technical_reference", scored_style
    assert selector_style == scored_style, {
        "scored_style": scored_style,
        "selector_style": selector_style,
    }
    samples = (
        "Step 1: open the settings page.",
        "- first item\n- second item\n- third item",
        "API parameter: value\nReturns: result",
        "Question: what do you think?",
        "A plain paragraph without special formatting.",
    )
    for text in samples:
        assert _style_bucket_from_text(text) == style_bucket_from_text(text), text
    alignment = _style_taxonomy_alignment_diagnostic(
        Counter({"general_prose": 2, "technical_reference": 1}),
        Counter({"general_prose": 2, "technical_reference": 1}),
        {
            "iterations": [
                {
                    "quota_diagnostics": {
                        "style_distribution_balance": {
                            "selected_bucket_counts_after": {
                                "general_prose": 2,
                                "technical_reference": 1,
                            }
                        }
                    }
                }
            ]
        },
    )
    assert alignment["aligned"] is True, alignment
    assert alignment["absolute_count_difference"] == 0, alignment
    print("[style-contract] Stage-B and full-chunk style taxonomy alignment: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
