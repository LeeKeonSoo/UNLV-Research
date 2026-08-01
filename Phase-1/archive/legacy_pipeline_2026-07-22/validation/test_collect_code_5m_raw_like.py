#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_code_5m_raw_like import eligibility_reason


def main() -> int:
    admission = {
        "allowed_spdx_licenses": ["MIT", "Apache-2.0"],
        "exclude_path_fragments": ["/tests/", "/vendor/"],
        "min_file_bytes": 256,
        "max_file_bytes": 262144,
    }
    eligible = {
        "content": "x = 1\n" * 100,
        "max_stars_repo_licenses": "['MIT']",
        "max_stars_repo_path": "src/example.py",
        "size": 600,
    }

    assert eligibility_reason(eligible, admission) is None
    assert eligibility_reason({**eligible, "max_stars_repo_licenses": "['GPL-3.0']"}, admission) == "license_not_allowed"
    assert eligibility_reason({**eligible, "max_stars_repo_path": "pkg/tests/example.py"}, admission) == "excluded_path"
    assert eligibility_reason({**eligible, "size": 100}, admission) == "file_too_small"
    assert eligibility_reason({**eligible, "content": ""}, admission) == "missing_content"
    print("[collect-code-5m-raw-like] admission rules: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
