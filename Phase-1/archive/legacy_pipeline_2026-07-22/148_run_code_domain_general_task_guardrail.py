#!/usr/bin/env python3
from __future__ import annotations

from archive.temporal_code.code_general_task_guardrail import (
    TASKS,
    _build_result_report,
    _completed_tasks,
    _covers_requested_tasks,
    _merge_lm_eval_results,
    _normalize_suite_status,
    main,
    run_missing,
    run_one,
)


if __name__ == "__main__":
    raise SystemExit(main())
