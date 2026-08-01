# LM Training Data Curation

This directory contains an auditable curation framework for language-model
training corpora. Its runtime boundary ends when it emits a curated dataset and
its decision trace. Continued pretraining and downstream benchmarks are an
external evaluation protocol, not selector inputs.

## Current Authority

Read these files in order:

1. `docs/framework_consistency_baseline.md` records observed behavior, known
   inconsistencies, and the frozen redesign order.
2. `HANDOFF.md` gives the current repository state and verification commands.
3. `docs/current_curation_framework.md` preserves detailed design context.

When these documents disagree, the consistency baseline wins. Candidate
designs, development reports, and historical experiments do not become active
runtime policy merely because their files exist.

## Research Boundary

The framework keeps the Core-Metric-Policy separation and the four public
Cores: Validity, Quality, Redundancy, and Coverage. The current implementation
does not support a claim that it measures intrinsic document Quality or
universally improves downstream performance.

Runtime policy must not consume benchmark scores, NLL, Utility, source
reputation, domain quotas, or a forced token budget. External experiments may
use those measurements only to validate a frozen policy after curation.

## Repository Map

- `run_curation.py`: public curation entry point.
- `ingestion/`: input adaptation and candidate contracts.
- `configs/`: runtime contracts plus candidate evidence configurations. See
  `configs/README.md` before treating a file as active.
- `protocols/`: development and external-evaluation protocols. They are not
  selector policy. See `protocols/README.md`.
- `validation/`: behavior and contract checks.
- `external_evaluation/`: post-curation training and benchmark tooling.
- `docs/`: current authority, design notes, development records, and historical
  analyses. See `docs/README.md`.
- `archive/`: legacy pipeline and pre-reduction snapshots. It is not imported by
  the current runtime.

The top-level Python surface still contains active runtime modules and
unpromoted candidate modules together. This is a known organization debt, not
evidence that every module is active. Physical relocation is deferred because
it changes imports and the active-surface test contract.

## Current Freeze

As of this baseline, repository organization and documentation may change, but
runtime policy, thresholds, scoring formulas, and selection behavior are frozen
until the user explicitly requests implementation. Do not run new training or
claim-producing experiments from this cleanup commit.

## Verification

From `Phase-1` in the `research` environment:

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

Passing these checks confirms the repository's existing contracts. It does not
close the semantic and lifecycle gaps listed in the consistency baseline.
