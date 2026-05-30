#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the old fixed-token probe interface.

Canonical utility measurement now lives in `utility.lm_probe` and exposes the
same public helper names.
"""

from __future__ import annotations

from utility.lm_probe import (  # noqa: F401
    DEFAULT_MODEL_NAME,
    SmallLMProbeContext,
    aggregate_probe_runs,
    build_probe_context,
    score_selected_records,
)
