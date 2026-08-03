# Framework Consistency Baseline

Status: observed baseline plus Block 9 development-ablation decision
Frozen on: 2026-08-01
Authority: this file is the sole status index during the consistency reset

Checkpoint intent: preserve a reviewed, reproducible snapshot of the observed
repository before any further policy implementation. This checkpoint does not
resolve the inconsistencies below and must not be cited as framework completion.

## Purpose

This document records what the repository does now, not what the project
intends to implement later. It prevents active runtime behavior, candidate
designs, external experiments, and historical evidence from being described as
one completed framework.

Until every exit condition in this document is satisfied:

- do not add or promote curation rules;
- do not start new confirmatory training runs;
- do not describe candidate Quality or Validity designs as runtime-active;
- do not describe a fixture pass as corpus-wide validity or downstream value;
- do not use prior benchmark outcomes to modify a confirmatory policy.

## Fixed Research Boundary

The project builds the curation layer between collected text and language-model
training data. The runtime ends when it materializes a curated JSONL corpus.
Training and benchmarks are external validation and may not change a frozen
curation output.

The public architecture retains four conceptual Cores:

1. Validity: determine whether the input is interpretable and processable
   without destroying its meaning.
2. Redundancy: remove repeated payload while retaining a deterministic linked
   representative.
3. Quality: reject only when a named policy has sufficient evidence that the
   unit is non-payload for the declared curation scope; otherwise retain by
   abstention.
4. Coverage: veto unexplained representative or residual-payload loss and
   report composition changes; it does not impose a target domain mix.

The runtime may not read Utility, NLL, benchmark outcomes, a target retention
fraction, or an implicit token budget. The framework does not currently provide
a universal intrinsic Quality score.

## Block 7 Runtime Integration

As of 2026-08-03, `run_curation.py` validates the redesign foundation before
reading corpus input. The preflight verifies the central manifest, typed object
registry, profile registry, compatibility bridge, legacy kernel, and Stage
permission identities. The run then records authorization tickets for Stage-A
Validity, Stage-B Redundancy, Stage-B Quality, and Stage-C Coverage.

This is an integration transition, not Policy promotion. The frozen selector
kernel remains behavior-compatible, both new v1 profiles are release-disabled,
and the near-duplicate and contrastive v1 policies remain `blocked`. Legacy
near-duplicate execution is explicitly reported as compatibility-only and must
be promoted or retired through development evidence rather than inherited by
the new profile.

## Block 8 Integrity Validation

As of 2026-08-03, `framework_release_validation.py` composes the existing Core
behavior audit with negative fail-closed scenarios and a deterministic
input-to-curated-output projection hash. Its frozen protocol is
`configs/framework_release_validation_v1.json`; its frozen result is
`validation/frozen_contracts/framework_release_validation_v1.json`.

All 39 Core behavior cases pass with 24 true positives, 15 true negatives, and
no false result or invariant failure. All nine implementation gates also pass,
including identity tamper detection, required threshold provenance, Stage/Core
authority, forbidden benchmark input rejection, provider non-authority,
unpromoted-profile rejection, profile monotonicity, and output equivalence.

This closes implementation-integrity Block 8, not scientific Policy promotion.
The report deliberately records `framework_release: blocked` because the new
profiles contain unpromoted Policies and both symmetric near-duplicate and
contrastive Quality remain blocked. Development calibration and independent
ablation are Block 9.

The Block 8 regression run passed 142/142 direct validation files with the
repository root on `PYTHONPATH`, GPU visibility disabled, and model-network
access offline. An invocation without the documented `PYTHONPATH` reproduced
the pre-existing Stage-C2 import-path failure; no curation behavior failed.

## Block 9 Development Ablation

As of 2026-08-03, `framework_policy_ablation.py` hash-verifies the admitted,
benchmark-excluded development corpus plus frozen Redundancy and Quality
evidence. Its protocol is `configs/framework_policy_ablation_v1.json`, and its
decision is frozen at
`validation/frozen_contracts/framework_policy_ablation_v1.json`.

The exact-text family Policy is `development_passed` from 2,400 correctly
linked exact copies, zero clean-control false merges, and zero representative
failures. This lifecycle is deliberately below `promoted` and does not make a
profile releasable.

Symmetric near-duplicate remains blocked because 860 observed candidate
relations are not a positive non-exact equivalence ground truth and cannot
identify a deletion threshold. Contrastive Quality remains blocked because the
three-role provider lacks a qualified reference distribution, assigned
background provider, shared Stage-A baseline, and route-specific empirical
effect bins. Neither policy emitted a threshold or gained runtime authority.

The resulting Hard profile is not development-ready, and Block 10 three-seed
natural-budget confirmation is not authorized. The bundle reads neither
benchmark outcomes nor Utility and does not mutate selector membership.

The Block 9 regression run passed 143/143 direct validation files and parsed
181/181 active config, protocol, and frozen-contract JSON files with GPU and
network access disabled.

## Observed Runtime

The only production entry point is `run_curation.py`.

```text
raw JSONL
  -> adapter
  -> Stage A: normalization and narrow text-integrity quarantine
  -> Stage B: chunking, invalid-result gate, exact post-Stage-A text deduplication
  -> Stage C: optional structural-family compaction and explicit artifact rules
  -> Coverage and composition audits
  -> curated JSONL
```

The Normal runtime is a reason-coded structural cleaner. Symmetric 0.95
near-duplicate compaction and four closed-set Quality rejection rules are
active. Positive-retention Quality evidence is not active, and Coverage has
veto-only materialization authority without ranking or deletion authority.

### Runtime Dependency Surface

| Layer | Runtime files |
| --- | --- |
| Entry and materialization | `run_curation.py`, `curation_artifacts.py` |
| Input and Stage A | `ingestion/input_adapter.py`, `ingestion/candidate_processing.py`, `ingestion/candidate_contract.py` |
| Stage C selection | `stage_c_selection.py`, `quality_retention.py`, `quality_decision_contract.py`, `quality_rule_evidence.py` |
| Hard development transforms | `hard_structural_runtime.py`, `inline_license_header_compaction.py`, `inline_license_comment_block_compaction.py`, `span_level_template_compaction.py` |
| Development-only web transform | `general_web_span_compaction.py` |
| Audit | `composition_audit.py`, `coverage_taxonomy.py`, `content_router.py`, `reason_code_audit.py` |
| Policy declarations hashed by runtime | `configs/curation_contract.json`, `configs/core_policy_registry.json`, `configs/policy_cards.json`, `configs/policy_profiles.json` |

Collectors, training scripts, benchmark runners, candidate estimators, and
paper tooling are not part of the input-to-output runtime.

## Core Status

| Core | Current runtime authority | Current evidence status | Honest status |
| --- | --- | --- | --- |
| Validity | narrow Stage-A quarantine and Stage-B invalid-result rejection | constructed fixtures pass; corpus precision/recall unknown | partially active |
| Redundancy | normalized exact duplicate and identical scaffold-family handling | exact behavior is executable; active-policy empirical status remains unvalidated | partially active |
| Quality | explicit generated/non-editable, license-only, empty-shell, and web-control artifact rules when enabled; otherwise `ABSTAIN_RETAIN` | positive route-conditioned evidence is inactive and all registered routes are missing or indeterminate | rejection-only and incomplete |
| Coverage | representative linkage, residual-payload, zero-survivor, and composition audit | executable audit fixtures pass | active audit, not a selector |

An inactive, candidate, diagnostic, or audit-only component must not be counted
as an active selection capability.

## Recent Runtime Evidence

The 2026-08-01 Normal replay used Qwen3-4B tokenization only for external
reporting. The runtime itself used a whitespace token proxy.

| Corpus | Raw Qwen tokens | Curated Qwen tokens | Reduction |
| --- | ---: | ---: | ---: |
| Legacy Code | 7,029,267 | 6,977,075 | 0.742% |
| The Stack v2 Python | 10,000,576 | 9,632,930 | 3.676% |
| OpenWebMath | 8,469,064 | 8,413,098 | 0.661% |
| FineWeb | 1,015,169 | 1,015,245 | -0.007% normalization/chunk-boundary delta |
| Common Crawl | 1,001,926 | 959,653 | 4.219% |

These results show that the current Normal policy has low opportunity on the
tested corpora. They do not prove that removed text is unnecessary or that the
retained corpus improves downstream training.

## Consistency Resolution Ledger

| ID | Original inconsistency | Implemented resolution | Status |
| --- | --- | --- | --- |
| C-01 | Profile ID did not determine runtime policy | Normal/Hard contain complete policies; run-local selector overrides fail closed; every report carries the effective policy and hash | Resolved |
| C-02 | Missing context could select destructive normalization | Missing normalization context now means exact preservation; explicit context is separate from PII context | Resolved |
| C-03 | Multiple populated text fields were concatenated | Exactly one populated declared field is accepted; ambiguous input raises an error | Resolved |
| C-04 | Residual threshold was described as a Stage-B gate | Renamed and moved to `stage_c.minimum_residual_chars` throughout current contracts | Resolved |
| C-05 | Authorized and enabled policies were conflated | Registry/profile distinguish authorization from enablement; the current Normal authorized set is fully enabled and the report emits the effective manifest | Resolved |
| C-06 | Executability and empirical validation were conflated | Lifecycle dimensions separately represent execution, profile enablement, and empirical evidence | Resolved |
| C-07 | Quality rejection and retained abstentions were conflated | Report separately counts explicit non-payload rejection, positive Quality keep, and abstain-retain | Resolved |
| C-08 | Coverage authority was described only as an audit | Coverage is a veto-only materialization invariant and cannot rank, delete, or quota-restore | Resolved |
| C-09 | Exact-duplicate representative depended on input order | Exact families use a stable digest and chunk-UID ordering with reversed-input fixture coverage | Resolved |
| C-10 | Whitespace proxies looked like training-token counts | Runtime labels them `whitespace_proxy_non_training`; exact tokenizer counts are external and require a declared tokenizer | Resolved |
| C-11 | Tests did not cover the contract gaps | Added negative tests for policy overrides, ambiguous fields, context preservation, deterministic representatives, and Coverage authority | Resolved |
| C-12 | No clean frozen baseline existed | Commit `2fd53ea` records the pre-implementation consistency baseline | Resolved |
| C-13 | Core inventory expected a removed candidate | Inventory contract now follows the frozen intended schema without restoring the unsupported rule | Resolved |
| C-14 | Frozen execution manifests pinned stale contract hashes | Affected manifests were re-fingerprinted against the preserved historical contract bytes | Resolved |

## Baseline Verification

The cleanup checkpoint was checked without network or GPU access:

- all 270 non-ignored JSON files parsed successfully;
- the active Python surface passed `compileall`;
- the eight handoff contract tests passed;
- 118 directly runnable validation files were exercised with the repository on
  `PYTHONPATH`: 116 passed and the two failures are C-13 and C-14;
- the five Stage-C2 tests that initially lacked the repository import path
  passed after `PYTHONPATH` was set;
- `pytest` is not installed in the `research` environment, so pytest-only
  collection was not run;
- staged diff, generated-artifact, large-file, and secret-pattern checks passed.

Secret-pattern matches in detector fixtures are intentional fake credentials.
Matches inside `task-artifact` and `risk-*` identifiers are regular-expression
false positives, not credentials.

## Post-Alignment Verification

The authorized implementation pass on 2026-08-01 completed C-01 through C-14.
With GPU access disabled and the repository on `PYTHONPATH`, all 120 directly
runnable validation files passed. Python `compileall` passed, all 131 JSON
contracts under `configs/` and `protocols/` parsed, and both frozen external
evaluation preflights passed their updated SHA-256 checks. This establishes
contract and fixture consistency only; it does not validate a universal Quality
estimator or domain-general downstream effectiveness.

## Artifact Classification

### Current Runtime

Only the files in the runtime dependency surface and their direct executable
tests may define current behavior.

### Candidate Design

The Content Router v2, Validity recovery, route-conditioned Quality, positive
provider evidence, aggressive structural candidates, and Hard transforms remain
candidate or development-only work. Their documentation may define proposed
contracts but cannot override this baseline or the observed runtime.

### External Evaluation

Files under `external_evaluation/` and confirmatory protocols evaluate frozen
outputs. They are not framework stages and have no curation authority.

### Historical Evidence

Prior fixed-fraction selectors, Utility experiments, proxy scorers, old stage
names, and superseded evaluation runs are historical evidence only. They may
explain a decision but cannot define current behavior.

## Redesign Blocks

### R1. Repository And Contract Freeze

- stop new rule and benchmark work;
- identify the minimal runtime and test surface;
- assign every other artifact to candidate, external, or historical scope;
- establish a clean reviewed Git baseline.

### R2. Input Preservation And Stage Semantics

- parse one unambiguous text field;
- preserve text by default;
- make every normalization reversible or explicitly audited;
- align Stage-A/B/C names, configuration fields, traces, and behavior.

### R3. Immutable Profiles And Policy Lifecycle

- make Normal and Hard resolve complete immutable policy sets;
- reject per-run policy drift;
- separate authorized, enabled, development-validated, and confirmatory states;
- emit the effective policy manifest and hash in every report.

### R4. Core Authority

- bind every removal or veto to exactly one Core and one named policy;
- separate artifact rejection from positive Quality evidence;
- keep Coverage as an enforced materialization invariant;
- remove active status from rules lacking their declared evidence gate.

### R5. Validation Reset

- add meaning-preservation and profile-resolution end-to-end tests;
- estimate rule opportunity and false-positive boundaries on disjoint raw-like corpora;
- freeze one development-selected policy before confirmatory evaluation;
- run Raw versus frozen profiles with natural token budgets and three seeds.

### R6. Release And Paper

- release only behavior represented by the frozen effective-policy manifest;
- state Framework, Policy, and Evidence claims separately;
- write the paper only after all prior blocks are closed.

## Exit Conditions For Reimplementation

Implementation work may resume only when:

1. this baseline and the runtime dependency list have been reviewed;
2. every top-level document points here for current status;
3. one immutable Normal profile contract is agreed;
4. input-preservation behavior is agreed;
5. each Core has a single declared authority and abstention behavior;
6. candidate and historical artifacts cannot be mistaken for active runtime;
7. no new training run is needed to decide basic runtime semantics.
