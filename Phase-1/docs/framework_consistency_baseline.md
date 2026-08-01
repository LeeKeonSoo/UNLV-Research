# Framework Consistency Baseline

Status: observed baseline before runtime redesign
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

## Observed Runtime

The only production entry point is `run_curation.py`.

```text
raw JSONL
  -> adapter
  -> Stage A: normalization and narrow text-integrity quarantine
  -> Stage B: chunking, invalid-result gate, normalized exact deduplication
  -> Stage C: optional structural-family compaction and explicit artifact rules
  -> Coverage and composition audits
  -> curated JSONL
```

The observed Normal runtime is a conservative structural cleaner. It is not an
aggressive SLM data selector. Positive-retention Quality evidence is not active,
near-duplicate compaction is disabled in the current frozen experiment
contracts, and Coverage is audit-only.

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

## Confirmed Inconsistencies

| ID | Inconsistency | Required resolution before redesign |
| --- | --- | --- |
| C-01 | `curation_mode: normal` resolves only a profile ID; run-contract booleans still determine the active rules | A profile must materialize one immutable complete policy configuration, and incompatible overrides must fail |
| C-02 | Missing `pii_context` defaults to `general`; code identified only through nested metadata can undergo whitespace-destructive normalization | Preserve input by default and make normalization context explicit, parsed, and meaning-preserving |
| C-03 | Multiple configured text fields are concatenated when more than one is populated | Select exactly one declared field or reject ambiguous input |
| C-04 | `minimum_chunk_chars` is reported in the Stage-B non-trigger boundary but is not a Stage-B rejection condition | Rename it to its residual-transform role or implement the declared gate; do not claim both |
| C-05 | Normal profile and Registry list policies that run contracts can disable; near-duplicate is declared active but frozen experiments disable it | Separate authorized policies from enabled policies and expose one effective-policy manifest per run |
| C-06 | Policies marked `active_structural` still carry `unvalidated_structural_policy` empirical status | Introduce distinct executable, development-validated, confirmatory-validated, and production states and enforce promotion atomically |
| C-07 | Quality output labels explicit artifact rejection as Quality while all retained chunks are abstentions | Report artifact rejection and positive Quality evidence separately |
| C-08 | Coverage passes as an audit but has no record-selection authority | Describe it as a materialization veto/audit, never as a fourth active selector |
| C-09 | Exact-duplicate representative choice depends on input order | Choose representatives by a deterministic corpus-order-independent key |
| C-10 | Runtime reports whitespace token proxy while training uses a model tokenizer | Report transformation-preservation counts separately and require a declared tokenizer for training-budget evidence |
| C-11 | Current tests pass despite C-01 through C-05 | Add end-to-end negative tests for effective profiles, ambiguous input, metadata-light code, and no-trigger byte preservation |
| C-12 | The working tree had no clean frozen implementation baseline | This cleanup checkpoint records the observed mismatch without changing behavior; all subsequent implementation must start from this snapshot |
| C-13 | `test_core_rule_inventory.py` expects `strong_generated_marker_candidate`, but the current inventory builder no longer emits that candidate | Freeze the intended inventory schema, then align the builder, registry, and test together; do not restore a rule merely to satisfy the old count |
| C-14 | The seven-benchmark execution manifest pins an outdated SHA-256 for `code_7m_pretraining_eligible_curation_v3_contract.json` | Decide which frozen artifact is authoritative and rebuild the manifest through its freeze procedure; never edit the digest alone |

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
