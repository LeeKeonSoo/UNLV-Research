#!/usr/bin/env python3
"""Regression checks for deterministic generated-file detection."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_change import generated_file_detection  # noqa: E402


def main() -> int:
    clean = generated_file_detection("src/runtime.py", "def run():\n    return True\n")
    assert clean == {"generated": False, "evidence": [], "status": "completed_heuristic_v1"}, clean

    filename = generated_file_detection("api/service_pb2.py", "class Service: pass")
    assert filename["generated"] is True, filename
    assert "generated_filename_pattern" in filename["evidence"], filename

    marker = generated_file_detection("src/client.py", "# Automatically generated. Do not edit.\n")
    assert marker["generated"] is True, marker
    assert "generated_content_marker" in marker["evidence"], marker

    unavailable = generated_file_detection("src/private.py", None, None)
    assert unavailable["status"] == "incomplete_no_text", unavailable
    print("[generated-file-detection] filename and content-marker evidence: pass")
    print("[generated-file-detection] unavailable content remains incomplete: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
