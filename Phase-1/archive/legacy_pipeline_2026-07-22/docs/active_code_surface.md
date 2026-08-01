# Active Code Surface

## Default Scope

Active framework work uses the root modules, `signals/`, `policy/`,
`ingestion/`, `paper_evidence/`, `validation/`, and the canonical registry in
`configs/canonical_execution_path_v1.json`.

`archive/` is excluded from default ripgrep search through `.ignore`. It holds
historical implementations retained only for compatibility wrappers and
full-scope reproducibility checks.

## Historical Access

Use `rg --no-ignore archive/` only when tracing a historical numbered command,
its compatibility wrapper, or a full-scope reproducibility failure.

## Current Canonical Path

The active paper-evidence path is defined by
`configs/canonical_execution_path_v1.json` and executed through
`run_canonical_paper_evidence.py`.

`docs/current_execution_inventory.md` lists canonical, active-support, and
historical compatibility scripts in the remaining numbered surface.

The canonical path excludes collection, GPU training, benchmark generation,
and historical temporal-code operations.
