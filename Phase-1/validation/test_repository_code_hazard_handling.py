#!/usr/bin/env python3
"""Regression checks for conservative, code-aware PII detection."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.candidate_processing import detect_hazards, normalize_text  # noqa: E402


def main() -> int:
    numeric_code = "timestamp_fixture = '2025 12 31 17 33 59'\nversion = 9381_893_601"
    general = detect_hazards(numeric_code)
    repository_code = detect_hazards(numeric_code, pii_context="repository_code")
    assert general["pii_detected"] is True, general
    assert repository_code["pii_detected"] is False, repository_code
    assert repository_code["diagnostics"]["phone_suppressed_count"] > 0, repository_code

    for text in (
        "support_phone = '+1 702 555 0199'",
        "Call the maintainer at (702) 555-0199 before release.",
        "Contact: 7025550199",
        "owner_email = 'private.person@example.com'",
    ):
        result = detect_hazards(text, pii_context="repository_code")
        assert result["pii_detected"] is True, (text, result)

    secret = detect_hazards(
        "api_key = abcdefghijklmnop",
        pii_context="repository_code",
    )
    assert secret["secret_detected"] is True, secret
    embedded_cookie = detect_hazards(
        "headers = {'cookie': 'session=abcdefghijklmnopqrstuvwxyz0123456789; auth=abcdefghijklmnopqrstuvwxyz0123456789'}",
        pii_context="repository_code",
    )
    assert embedded_cookie["secret_detected"] is True, embedded_cookie
    code = "def compare(left, right):\n    return left < right\n"
    normalized = normalize_text(code, context="repository_code")
    assert normalized["text"] == code, normalized
    assert normalized["transformations"] == [], normalized
    print("[code-pii-detection] numeric code false positives suppressed: pass")
    print("[code-pii-detection] high-confidence phone, email, and secret detection retained: pass")
    print("[code-pii-detection] repository code whitespace and operators preserved: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
