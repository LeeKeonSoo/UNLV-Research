#!/usr/bin/env python3
"""Validate paired NLL analysis helpers without model execution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "184_evaluate_redundancy_proxy_nll.py"
    spec = importlib.util.spec_from_file_location("redundancy_proxy_nll", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    positive = module._paired_summary([0.004, 0.005, 0.006], 2.919986)
    assert positive["mean"] == 0.005
    assert positive["one_sided_95_lower"] > 0.002
    assert positive["positive_seed_count"] == 3

    noninferior = module._paired_summary([-0.001, 0.0, 0.0005], 2.919986)
    assert noninferior["one_sided_95_upper"] <= 0.002
    assert noninferior["nonpositive_seed_count"] == 2

    noisy = module._paired_summary([-0.003, 0.0, 0.004], 2.919986)
    assert noisy["one_sided_95_lower"] < 0.002
    assert noisy["paired_mde_95"] > abs(noisy["mean"])
    print("[redundancy-proxy-nll-analysis] paired CI and MDE rules: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
