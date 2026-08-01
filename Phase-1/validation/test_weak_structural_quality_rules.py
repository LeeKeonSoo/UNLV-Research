#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_c_selection import select_chunks


def _chunk(chunk_uid: str, text: str) -> dict[str, object]:
    return {
        "chunk_uid": chunk_uid,
        "stage_a_record_id": chunk_uid,
        "text": text,
        "token_proxy": len(text.split()),
    }


def main() -> int:
    # Given: a complete HTML document with no visible textual payload.
    empty_shell = _chunk(
        "empty-shell",
        "<!doctype html><html><head><meta charset=\"utf-8\"><title></title></head><body></body></html>",
    )
    # When: the explicit empty-shell policy is enabled.
    selected, removed, audit = select_chunks(
        [empty_shell],
        {"structural_artifact_rules": {"empty_html_shell": True}},
    )
    # Then: the shell is removed with a reason code.
    assert not selected
    assert [row["stage_c_selection"]["removed_reason"] for row in removed] == ["empty_html_shell"]
    assert audit["empty_html_shell_removed_chunks"] == 1

    # Given: a navigation/cookie-control fragment with no explanatory prose.
    web_chrome = _chunk(
        "web-chrome",
        "Cookie preferences\nAccept all\nReject all\nManage preferences\nPrivacy policy\nTerms of service",
    )
    # When: the explicit web-chrome policy is enabled.
    selected, removed, audit = select_chunks(
        [web_chrome],
        {"structural_artifact_rules": {"web_chrome_only_chunk": True}},
    )
    # Then: it is removed, without reading metadata or a quality score.
    assert not selected
    assert [row["stage_c_selection"]["removed_reason"] for row in removed] == ["explicit_web_chrome_only_chunk"]
    assert audit["web_chrome_only_removed_chunks"] == 1

    # Given: nearby useful text containing the same lexical markers.
    explanatory_cookie_text = _chunk(
        "cookie-explanation",
        "This tutorial explains how a browser stores cookie preferences and how a user can revoke consent without losing application state.",
    )
    substantive_html = _chunk(
        "substantive-html",
        "<!doctype html><html><body><article><h1>Derivative rules</h1><p>The derivative of a sum is the sum of derivatives.</p></article></body></html>",
    )
    # When: both rules are enabled.
    selected, removed, _ = select_chunks(
        [explanatory_cookie_text, substantive_html],
        {"structural_artifact_rules": {"empty_html_shell": True, "web_chrome_only_chunk": True}},
    )
    # Then: the nearby useful chunks remain intact.
    assert {row["chunk_uid"] for row in selected} == {"cookie-explanation", "substantive-html"}
    assert not removed
    assert {row["quality_retention_decision"]["decision"] for row in selected} == {"abstain_retain"}

    print("[weak-structural-quality] explicit empty shells and web chrome only: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
