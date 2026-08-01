#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from quality_rule_evidence import EXPLICIT_GENERATED_ARTIFACT_RE

    cases = json.loads((ROOT / "validation" / "fixtures" / "explicit_generated_artifact_cases.json").read_text(encoding="utf-8"))
    observed = {case["id"]: bool(EXPLICIT_GENERATED_ARTIFACT_RE.search(case["text"])) for case in cases}
    expected = {case["id"]: case["expected"] for case in cases}
    assert observed == expected
    print("[explicit-generated-artifact] labeled fixture precision gate: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
