# Handoff

## Start Here

The sole current-status authority is
`docs/framework_consistency_baseline.md`. Read it before changing policy or
running new experiments. `README.md` is the repository overview and
`docs/current_curation_framework.md` is detailed design context.

## Cleanup Baseline

This checkpoint is an organization and consistency freeze. It does not alter
runtime selection behavior. The user explicitly requested that no policy or
implementation work continue until separately authorized.

Current repository facts:

- The public framework boundary is corpus input to curated dataset plus an
  auditable decision trace.
- Continued pretraining, NLL, and benchmark execution are external validation.
- Runtime must not read Utility, benchmark results, source reputation, domain
  quotas, or a forced token budget.
- The public Cores remain Validity, Quality, Redundancy, and Coverage.
- Current observed selection is narrower than the intended four-Core design.
  Quality has no promoted positive provider and Coverage is audit-only.
- Candidate and historical files must not be described as active policy.

## Known Blocking Inconsistencies

The baseline tracks the full list as `C-01` through `C-12`. The highest-impact
items are:

1. The Normal profile ID does not completely determine runtime behavior; run
   configuration booleans still control rules.
2. Missing `pii_context` can select general normalization for code-like text,
   risking whitespace-sensitive corruption.
3. Lifecycle labels can say `active` even when runtime authority is absent.
4. The four-Core public model is not fully represented by active behavior.
5. Exact-duplicate representative choice depends on input order.
6. Audit token proxies and exact tokenizer counts are not consistently
   separated in reports.
7. The Core rule inventory test and the frozen seven-benchmark contract hash
   are stale relative to their current artifacts (`C-13` and `C-14`).

These are recorded defects, not tasks completed by this cleanup.

## File Authority

- `docs/README.md` classifies documentation as authoritative, candidate,
  external-evaluation, historical, or template material.
- `configs/README.md` explains that configuration presence is not runtime
  activation.
- `protocols/README.md` separates selector policy from development and
  confirmatory evaluation.
- `archive/README.md` defines the legacy boundary.

The top-level Python surface remains mixed because moving modules would change
imports and tests. Use registry linkage and runtime call paths, not file
location alone, to determine authority.

## Verification Order

Run from `Phase-1`:

```powershell
conda run -n research python validation\test_active_surface.py
conda run -n research python validation\test_curation_contract.py
conda run -n research python validation\test_candidate_processing.py
conda run -n research python validation\test_curation_runtime.py
conda run -n research python validation\test_policy_profile_contract.py
conda run -n research python validation\test_core_policy_runtime_linkage.py
conda run -n research python validation\test_core_behavior_audit_v3.py
conda run -n research python validation\test_source_contract.py
```

Also run `git diff --check`, a secret scan, and a staged large-file scan before
committing. Generated datasets, model caches, benchmark outputs, rendered
papers, and local work directories must remain ignored.

At this checkpoint, the eight commands above pass. A broader direct-run sweep
passes 116 of 118 validation files; the two failures are deliberately left
unfixed and documented as `C-13` and `C-14`. `pytest` is not installed in the
`research` environment.

## Next Authorized Work

Only after the user requests implementation, follow redesign blocks `R1` to
`R6` in the consistency baseline. Do not tune Quality formulas or add stronger
rules before the runtime contract, normalization safety, lifecycle semantics,
and measurement vocabulary are fixed.
