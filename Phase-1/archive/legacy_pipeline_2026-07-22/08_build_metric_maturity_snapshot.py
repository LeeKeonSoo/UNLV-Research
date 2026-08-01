#!/usr/bin/env python3
"""Build the metric maturity snapshot from existing pipeline outputs."""

from __future__ import annotations

from reports.metric_maturity import build_metric_maturity_snapshot


def main() -> int:
    path = build_metric_maturity_snapshot()
    print(f"[08] metric maturity snapshot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
