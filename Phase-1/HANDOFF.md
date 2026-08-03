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
- Block 7 now loads and hash-verifies the central framework manifest, typed
  object registry, profile registry, runtime bridge, and Stage permissions at
  the start of every materialization. Each run emits four authorization tickets
  for Validity, Redundancy, Quality, and Coverage.
- Block 8 now produces one deterministic release-validation bundle. The frozen
  report passes all Core fixture and implementation-integrity gates while
  keeping the scientific framework release blocked.
- Block 9 now produces a hash-linked development-ablation decision bundle.
  Exact-text family removal is `development_passed`; symmetric near-duplicate
  and contrastive Quality remain `blocked`, so Hard and Block 10 are not
  authorized.
- The selector kernel still executes the frozen legacy-compatible Normal/Hard
  behavior. No blocked v1 Policy was promoted by the bridge, and both new v1
  profiles remain release-disabled.
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

- `validation/frozen_contracts/framework_release_validation_v1.json` is the
  frozen Block 8 bundle. It covers 39 labeled, false-positive, metamorphic, and
  adversarial Core cases: 24 true positives, 15 true negatives, zero false
  positives, zero false negatives, and zero behavior-invariant failures.
- Nine implementation gates pass: foundation hash chain, kernel tamper
  detection, threshold provenance completeness, Stage/Core authority, runtime
  forbidden inputs, provider non-authority, unpromoted-profile rejection,
  Normal/Hard retained-set monotonicity, and curated-output equivalence.
- `implementation_integrity` is `passed`, but `framework_release` is `blocked`.
  The blockers are the unpromoted profile inventory plus blocked near-duplicate
  and contrastive Quality Policies. This is intentional and prevents a fixture
  pass from being reported as downstream effectiveness.

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

## Block 9 Evidence Status

- `validation/frozen_contracts/framework_policy_ablation_v1.json` is the frozen
  Block 9 decision bundle. It verifies the admitted development corpus and
  consumes no benchmark outcome or Utility input.
- `redundancy.exact_text_family` is now `development_passed`: 2,400 injected
  exact copies were linked with zero clean-control false merges and zero
  representative failures. This is development evidence, not downstream
  effectiveness or release promotion.
- `redundancy.symmetric_near_duplicate_candidate` remains `blocked`. The audit
  found 860 candidate relations but has no labeled non-exact equivalence set
  from which a deletion threshold can be identified; no threshold was emitted.
- `quality.contrastive_alignment_candidate` remains `blocked`. The frozen run
  scored 1,500 records, but a qualified reference distribution, background
  provider, common Stage-A baseline, and empirical route effect bins are still
  missing; no scalar or threshold was emitted.
- Consequently `hard_profile_development_ready` and `block_10_authorized` are
  both false. No runtime activation or selector membership was changed.
- The Block 9 regression run passed 143/143 direct validation files and parsed
  181/181 active config, protocol, and frozen-contract JSON files with GPU and
  network access disabled.

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
conda run -n research python validation\test_framework_release_validation_v1.py
conda run -n research python validation\test_framework_policy_ablation_v1.py
conda run -n research python validation\test_source_contract.py
```

Also run `git diff --check`, a secret scan, and a staged large-file scan before
committing. Generated datasets, model caches, benchmark outputs, rendered
papers, and local work directories must remain ignored.

`pytest` is not installed in the `research` environment, so validation files
are run directly with the repository on `PYTHONPATH`. The 2026-08-01 alignment
pass completed 120/120 direct validation files, Python compileall, and 131/131
current config/protocol JSON parses with GPU and network use disabled.

The 2026-08-03 Block 8 pass completed 142/142 direct validation files with the
repository on `PYTHONPATH`, GPU visibility empty, and Hugging Face/Transformers
offline. The first unqualified invocation reproduced the known Stage-C2 import
failure; the authoritative run includes the documented repository
`PYTHONPATH` and passed in full.

## Next Authorized Work

Follow `docs/framework_research_contract_v1.md` for redesign decisions. The
redesign foundation through Block 7 is implemented. The production entry point
consumes the root manifest, typed Core-Metric-Policy-Method-Provider objects,
Stage permissions, and the compatibility bridge before it reads corpus input.
The bridge preserves the frozen selector output and does not activate blocked
v1 policies.

Block 9 is complete, but its result does not authorize Block 10. The next
authorized work is to construct a positive, non-exact equivalence calibration
set for near-duplicate deletion and to qualify the three-role contrastive
provider with one common Stage-A baseline and route-specific effect bins.
Only a new frozen Block 9 decision that passes those gates may freeze Hard and
authorize three-seed natural-budget confirmatory evaluation.
