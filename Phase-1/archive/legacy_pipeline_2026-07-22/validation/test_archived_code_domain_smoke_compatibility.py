#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


MODULES = (
    ("136_fetch_known_high_quality_python_reference_pool", "archive.temporal_code.code_reference_pool", "fetch"),
    ("137_freeze_code_domain_equal_token_training_arms", "archive.temporal_code.code_equal_token_arms", "freeze"),
    ("139_build_code_domain_qlora_smoke_report", "archive.temporal_code.code_qlora_smoke_report", "build"),
    ("140_freeze_code_domain_development_plan", "archive.temporal_code.code_development_plan", "freeze"),
)


def main() -> int:
    for wrapper_name, archive_name, callable_name in MODULES:
        wrapper = importlib.import_module(wrapper_name)
        archived = importlib.import_module(archive_name)
        assert getattr(wrapper, callable_name) is getattr(archived, callable_name)
        assert wrapper.main is archived.main
    plan = importlib.import_module("archive.temporal_code.code_development_plan")
    assert plan._resolve("configs/code_domain_development_plan_qwen3_4b_v1.json") == (
        PROJECT_DIR / "configs" / "code_domain_development_plan_qwen3_4b_v1.json"
    )
    print("[archived-code-domain-smoke] root wrappers and project paths: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
