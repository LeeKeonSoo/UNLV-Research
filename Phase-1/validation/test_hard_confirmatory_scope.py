#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import resolve_curation_mode


def main() -> int:
    mode = resolve_curation_mode("hard", execution_scope="confirmatory")
    assert mode["mode"] == "hard"
    assert mode["profile_id"] == "hard_structural_v1"
    assert mode["authorization"] == "confirmatory_only_pending_external_decision"
    assert mode["effective_policy_sha256"]
    try:
        resolve_curation_mode("hard", execution_scope="production")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Hard production use must remain fail-closed during confirmatory evaluation.")
    print("[hard-confirmatory-scope] Hard confirmatory-only authorization: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
