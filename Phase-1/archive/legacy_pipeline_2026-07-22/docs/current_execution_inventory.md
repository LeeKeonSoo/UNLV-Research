# Current Execution Inventory

## Canonical Paper Evidence

Run `python run_canonical_paper_evidence.py --execute` for the current paper
evidence path. The canonical numbered compatibility entrypoints are:

- `190_run_paper_claim_release_gate.py`
- `211_build_code_paper_evidence_report.py`
- `213_build_final_paper_evidence_table.py`
- `218_build_paper_claim_consistency_audit.py`
- `219_build_domain_composition_audit.py`
- `220_build_coverage_domain_mix_audit.py`
- `221_build_stage_b_policy_contract_audit.py`

`222_build_canonical_execution_registry.py` validates this registry and stays
active.

## Active Support Evidence

Keep these root scripts active because retained tests and current framework or
paper boundaries consume their reports:

- `192`--`199`: Core claim, Stage-0, Stage-C, confirmatory, package, table,
  reproducibility, and raw-vs-curated readiness reports.
- `210_build_production_readiness_gate_report.py`,
  `212_build_stage0_release_blocker_report.py`, and
  `213_build_record_disposition_audit_report.py`: release and disposition
  evidence.
- `214`--`217`: current Math selector-v3 materialization and Stage-C summary.
- `223_build_hf_mixed_corpus_retest_protocol.py`: frozen next-cycle retest
  protocol.

## Historical Compatibility Evidence

These scripts remain reproducible historical evidence but are not in the
canonical runner:

- `191_score_core_metrics_parallel.py`
- `200_build_code_domain_block3_benchmark_report.py`
- `201`--`209`: prior Math acquisition, v2 selector, and natural-budget
  materialization/reporting path.
- `210_build_math_failure_fixture_contract.py`: historical Math failure
  fixture contract.

Do not use historical compatibility evidence to upgrade a paper or framework
claim without a current-framework rerun.

## Execution Rule

Use the canonical runner for current paper evidence. Run an active support
script only when regenerating its named report. Access `archive/` only for a
historical command or full-scope reproducibility investigation.
