#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_rule_development_matrix import run_quality_matrix


def test_quality_matrix_isolates_active_rules_and_keeps_web_candidate_inert() -> None:
    rows = [
        {"chunk_uid": "article", "stage_a_record_id": "article", "text": "A substantive article explains reproducible curation experiments in detail."},
        {"chunk_uid": "chrome", "stage_a_record_id": "chrome", "text": "Cookie Preferences\nAccept All\nReject All\nManage Preferences"},
        {"chunk_uid": "web", "stage_a_record_id": "web", "text": "A substantive archive article explains access policy.\n\nCookie Preferences\nAccept All\nReject All\nManage Preferences"},
        {"chunk_uid": "error-nav", "stage_a_record_id": "error-nav", "text": "404 Page Not Found\nHome\nAbout\nContact\nSearch"},
        {"chunk_uid": "urls", "stage_a_record_id": "urls", "text": "https://a.example\nhttps://b.example\nhttps://c.example\nhttps://d.example\nhttps://e.example"},
    ]

    report = run_quality_matrix(rows, minimum_chunk_chars=40, token_counter=lambda text: len(text.split()))

    assert set(report["arms"]) == {
        "baseline",
        "explicit_generated_artifact",
        "license_comment_only",
        "empty_html_shell",
        "web_chrome_only",
        "all_active_quality",
        "explicit_error_navigation_only_candidate",
        "url_directory_only_candidate",
        "web_control_span_candidate",
        "cumulative_quality_candidate",
    }
    assert report["arms"]["web_chrome_only"]["summary"]["curated_chunks"] == 4
    assert report["arms"]["explicit_error_navigation_only_candidate"]["summary"]["curated_chunks"] == 4
    assert report["arms"]["url_directory_only_candidate"]["summary"]["curated_chunks"] == 4
    assert report["arms"]["web_control_span_candidate"]["summary"]["transformed_span_count"] == 1
    assert report["arms"]["web_control_span_candidate"]["runtime_active"] is False
    assert report["arms"]["web_control_span_candidate"]["coverage"]["passed"] is True
    assert report["arms"]["cumulative_quality_candidate"]["summary"]["curated_chunks"] == 2
    assert all(arm["selector_boundary"]["benchmark_outcomes_read"] is False for arm in report["arms"].values())


if __name__ == "__main__":
    test_quality_matrix_isolates_active_rules_and_keeps_web_candidate_inert()
    print("[quality-rule-development-matrix] isolated Quality arms: pass")
