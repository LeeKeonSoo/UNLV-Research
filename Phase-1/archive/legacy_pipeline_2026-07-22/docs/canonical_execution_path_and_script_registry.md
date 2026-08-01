# Canonical Execution Path and Script Registry

## Purpose

The repository now contains many numbered scripts because the framework has
gone through repeated experiment, repair, and audit cycles. The canonical
execution registry separates the current paper-evidence rebuild path from older
or heavier scripts.

This registry does not delete historical work. It defines the small set of
commands that rebuild the current claim-facing evidence package from existing
Stage-C reports and guardrail artifacts.

## Canonical Path

The current lightweight path is:

1. `211_build_code_paper_evidence_report.py`
2. `227_build_code_livecodebench_pilot_summary.py`
3. `213_build_final_paper_evidence_table.py`
4. `190_run_paper_claim_release_gate.py`
5. `219_build_domain_composition_audit.py`
6. `220_build_coverage_domain_mix_audit.py`
7. `221_build_stage_b_policy_contract_audit.py`
8. `218_build_paper_claim_consistency_audit.py`

The path is configured in `configs/canonical_execution_path_v1.json` and
audited by `222_build_canonical_execution_registry.py`.

The numbered scripts remain stable compatibility entry points. Their active
implementations live together in `paper_evidence/`; historical scripts are not
imported by this package.

Run every listed step from the single manifest-driven entry point:

```powershell
python run_canonical_paper_evidence.py --execute
```

Exit code `2` means that a required claim gate blocked the evidence package;
the runner completed and recorded that decision.

## Boundary

This is a paper-evidence rebuild path, not a full experiment runner. It must not
include raw-data acquisition, dataset collection, QLoRA/GPU training, benchmark
sample generation, or production release execution.

Heavy upstream reports remain support reports. They are required inputs for the
current evidence package, but they are not rerun by the lightweight canonical
path.

## Verification

Run:

```bash
python 222_build_canonical_execution_registry.py
python validation/test_canonical_execution_registry.py
```

The latest report is:

- `outputs/validation/canonical_execution_registry_report.json`
- `outputs/validation/canonical_execution_registry_report.md`

Current status: `canonical_execution_registry_passed`.
