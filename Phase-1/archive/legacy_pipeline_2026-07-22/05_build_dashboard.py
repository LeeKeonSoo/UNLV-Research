#!/usr/bin/env python3
"""Canonical entrypoint: build dashboard for generic data evaluation."""

from __future__ import annotations

from reports.dashboard import build_dashboard


def main() -> int:
    path = build_dashboard()
    print(f"[05] dashboard: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
