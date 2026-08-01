#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from python_code_evidence_audit import analyze_python_source_records

    report = analyze_python_source_records(
        [
            {
                "record_id": "implementation",
                "text": "def add(left, right):\n    return left + right\n",
                "language": {"code": "python", "version": "3.11"},
            },
            {
                "record_id": "broken",
                "text": "def add(:\n    pass\n",
                "language": {"code": "python", "version": "3.11"},
            },
            {
                "record_id": "stub-only",
                "text": "def load(value):\n    ...\n\nclass Adapter:\n    def connect(self):\n        pass\n",
                "language": {"code": "python", "version": "3.11"},
            },
            {
                "record_id": "exception",
                "text": "class ValidationError(Exception):\n    pass\n",
                "language": {"code": "python", "version": "3.11"},
            },
            {
                "record_id": "version-unresolved",
                "text": "def broken(:\n    pass\n",
                "language": {"code": "python"},
            },
            {"record_id": "other", "text": "const value = 1;", "language": {"code": "javascript"}},
        ]
    )

    assert report["status"] == "candidate_evidence_only_not_a_runtime_selection_policy"
    assert report["counts"] == {
        "python_records": 5,
        "syntax_error": 1,
        "non_executable_stub_candidate": 1,
        "retained_without_candidate_evidence": 2,
        "version_unresolved_not_evaluated": 1,
        "non_python": 1,
    }
    assert report["candidate_record_ids"] == {
        "python_syntax_error_source_candidate": ["broken"],
        "python_non_executable_stub_source_candidate": ["stub-only"],
    }
    print("[python-code-evidence-audit] candidate-only structural code evidence: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
