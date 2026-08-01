#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import validate_quality_candidate_scope


def main() -> int:
    selection = {
        "structural_artifact_rules": {
            "explicit_error_navigation_only_chunk_candidate": True,
            "url_directory_only_chunk_candidate": True,
        },
        "quality_span_candidate_rules": {"web_control_and_url_directory": True},
    }
    assert validate_quality_candidate_scope(selection, "development") == [
        "explicit_error_navigation_only_chunk_candidate",
        "url_directory_only_chunk_candidate",
        "web_control_and_url_directory_span_candidate",
    ]
    try:
        validate_quality_candidate_scope(selection, "production")
    except RuntimeError as error:
        assert "development" in str(error)
    else:
        raise AssertionError("Candidate Quality policies must fail closed outside development.")

    print("[quality-candidate-scope] development-only execution boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
