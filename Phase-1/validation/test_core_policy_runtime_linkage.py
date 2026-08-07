#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.candidate_processing import STAGE_A_POLICY_REASON_CODES
from run_curation import STAGE_B_POLICY_REASON_CODES


def _active_registry_reason_codes() -> dict[str, frozenset[str]]:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    return {
        str(policy["id"]): frozenset(str(reason) for reason in policy["reason_codes"])
        for policy in registry["policies"]
        if policy["status"] == "active" and policy["reason_codes"]
    }


def main() -> int:
    registry_reason_codes = _active_registry_reason_codes()
    runtime_reason_codes = {
        **STAGE_A_POLICY_REASON_CODES,
        **STAGE_B_POLICY_REASON_CODES,
    }
    assert registry_reason_codes == runtime_reason_codes
    print("[core-policy-runtime-linkage] active Registry and runtime reason codes: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
