#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_web_span_compaction import build_plan, materialize_candidate_plan


def test_web_controls_and_url_directory_are_removed_without_dropping_article_payload() -> None:
    article = "This article explains how local archives preserve historical records for community researchers."
    rows = [
        {
            "chunk_uid": "web::article",
            "text": f"{article}\n\nCookie Preferences\nAccept All\nReject All\nManage Preferences\n\nhttps://example.com/a\nhttps://example.com/b\nhttps://example.com/c",
        }
    ]

    token_counter = lambda text: len(text.split())
    plan = build_plan(rows, minimum_residual_chars=40, token_counter=token_counter)
    result = materialize_candidate_plan(rows, plan, token_counter=token_counter)

    assert plan["candidate_span_removals"] == 2
    assert result["records"][0]["text"] == article
    assert {item["reason_code"] for item in result["transformations"]} == {
        "web_control_span_removed",
        "url_directory_span_removed",
    }
    assert sum(item["span_token_proxy"] for item in result["transformations"]) == (
        token_counter(rows[0]["text"]) - token_counter(article)
    )


def test_privacy_policy_article_is_retained() -> None:
    text = (
        "Privacy policy changes can affect how researchers access historical records.\n\n"
        "This article compares consent rules across several public archives and explains the tradeoffs."
    )

    plan = build_plan([{"chunk_uid": "reference::privacy", "text": text}], minimum_residual_chars=40)

    assert plan["candidate_span_removals"] == 0


def test_dialogue_is_protected_from_transactional_line_compaction() -> None:
    text = "Ava: Did you subscribe to the museum newsletter?\nBen: Not yet, but I want the event schedule.\nAva: I will send it after class."

    plan = build_plan([{"chunk_uid": "dialogue::chat", "text": text}], minimum_residual_chars=40)

    assert plan["candidate_span_removals"] == 0


if __name__ == "__main__":
    test_web_controls_and_url_directory_are_removed_without_dropping_article_payload()
    test_privacy_policy_article_is_retained()
    test_dialogue_is_protected_from_transactional_line_compaction()
    print("[general-web-span-compaction] payload-preserving web structure: pass")
