# Handoff

## Start Here

The sole current-status authority is
`docs/framework_consistency_baseline.md`. Read it before changing policy or
running new experiments. `README.md` is the repository overview and
`docs/current_curation_framework.md` is detailed design context. Quality and
Coverage semantics are fixed in `docs/quality_coverage_formal_definition.md`.
The frozen target for the next framework version is
`docs/framework_research_contract_v1.md`. It is the redesign authority, not a
statement that the current runtime already implements the target.

## Current Alignment

The cleanup baseline was followed by an authorized contract-alignment pass.
`normal` now resolves to a complete immutable policy; incompatible run-local
policy overrides fail closed.

Current repository facts:

- The public framework boundary is corpus input to curated dataset plus an
  auditable decision trace.
- Continued pretraining, NLL, and benchmark execution are external validation.
- Runtime must not read Utility, benchmark results, source reputation, domain
  quotas, or a forced token budget.
- The public Cores remain Validity, Quality, Redundancy, and Coverage.
- Quality has no promoted positive provider. Normal now enables the four
  closed-set non-payload rejection rules for explicit generated-and-do-not-edit
  artifacts, license-comment-only chunks, empty HTML shells, and cookie-control
  chrome-only chunks; all other cases abstain and retain.
- Coverage is a veto-only materialization invariant. It cannot rank, delete,
  restore by quota, or target a composition, but unexplained representative or
  residual-payload loss aborts output.
- Candidate and historical files must not be described as active policy.

## Block 8 Evidence Status

- E1 corpus admission is complete: Code, Math, and General development sources
  are benchmark-excluded and disjoint from frozen confirmatory references.
- E2 Redundancy behavior evidence is complete: 1,200/1,200 injected exact
  families and 2,400/2,400 exact copies were recovered with zero clean-control,
  perturbation, or cross-parent safe merge. Near, containment, and repeated-span
  relations remain candidate-only, and runtime activation remains false.
- The development preflight now hash-verifies the E3 Quality registry and
  report rather than trusting a readiness boolean. The evidence boundary is
  implemented, but the empirical gate remains blocked: Code, Math, and General
  route transfer are not ready, no active Quality provider exists, no measured
  effect bins exist, and no empirical common-baseline artifact exists. The two
  preflight blockers remain `quality_gate_not_ready` and
  `coverage_gate_not_ready`.
- E3b now has a replaceable target/reference scorer. Qwen3-4B-Base and
  Qwen3-8B-Base are the first audit pair, not framework dependencies. Their
  frozen development run joined 1,500 records with zero mismatch across 300
  exact copies, and their four native tokenizer files are byte-identical.
- The E3b result is still blocked and cannot delete data. Boilerplate frequently
  reduced absolute NLL and entropy, while excess NLL did not provide a stable
  route-general boundary. The audit records nine blockers: unvalidated int8,
  unverifiable provider-training disjointness, no common baseline, only two
  source groups per route, and no empirical effect bins for Code, Math, or
  General.

## Resolved Consistency Defects

The baseline tracks `C-01` through `C-14`. The implementation pass resolved:

1. immutable Normal/Hard policy manifests and override rejection;
2. byte-preserving default ingestion and ambiguous text-field rejection;
3. explicit executable/profile/empirical lifecycle dimensions;
4. deterministic exact-duplicate representative selection;
5. separate whitespace-proxy and exact-tokenizer measurement contracts;
6. distinct Quality rejection, positive keep, and abstain reporting;
7. Coverage materialization authority and executable fixtures;
8. stale inventory and frozen-manifest hashes.

The unresolved scientific work is Quality estimator validation and broader
Coverage taxonomy/threshold validation, not a hidden runtime inconsistency.

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

`pytest` is not installed in the `research` environment, so validation files
are run directly with the repository on `PYTHONPATH`. The 2026-08-01 alignment
pass completed 120/120 direct validation files, Python compileall, and 131/131
current config/protocol JSON parses with GPU and network use disabled.

## Next Authorized Work

Follow `docs/framework_research_contract_v1.md` for redesign decisions. The
next block is a single machine-readable framework schema that expresses that
contract without activating new removal rules. Do not tune Quality formulas or
add stronger rules before the schema, object contracts, stage permissions, and
measurement vocabulary are implemented and tested.
